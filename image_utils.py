# Import stuff
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as ur
from scipy.signal import convolve2d

def gaussian_psf(fwhm,patch_size,pixel_size):
    '''
    Return 2D array of a symmetric Gaussian PSF
    
    Required inputs:
    fwhm = FWHM in arcsec (4 * ur.arcsec)
    patch_size = Axis sizes of returned galaxy patch in pixels (15,15)
    pixel_size = Angular size of pixel (6 * ur.arcsec)
    '''
    x = np.linspace(-(patch_size[0] // 2), patch_size[0] // 2, patch_size[0])
    y = np.linspace(-(patch_size[1] // 2), patch_size[1] // 2, patch_size[1])
    x, y = np.meshgrid(x,y)
    
    sigma_pix = fwhm / (2. * np.sqrt(2 * np.log(2)) * pixel_size)
    psf = np.exp(-(x**2 + y**2) / (2 * sigma_pix**2)) / (2 * np.pi * sigma_pix**2)

    return psf

def sim_galaxy(patch_size,pixel_size,gal_type=None,gal_params=None):
    '''
        Return 2D array of a Sersic profile to simulate a galaxy
        
        Required inputs:
        patch_size = Axis sizes of returned galaxy patch in pixels (15,15)
        pixel_size = Angular size of pixel (6 * ur.arcsec)
        
        Optional inputs:
        gal_type = String that loads a pre-built 'average' galaxy or allows custom definition
    '''
    from astropy.modeling.models import Sersic2D
    
    x = np.linspace(-(patch_size[0] // 2), patch_size[0] // 2, patch_size[0])
    y = np.linspace(-(patch_size[1] // 2), patch_size[1] // 2, patch_size[1])
    x, y = np.meshgrid(x,y)
    
    # Takes either a keyword or Sersic profile parameters
    # Parameters currently meaningless
    if gal_type == 'spiral':
        amplitude = 0.1 # Surface brightness at r_eff in count rate
        r_eff = 20
        n = 1
        theta = 0
        ellip = 0.5
        x_0, y_0 = 20, 0
    elif gal_type == 'elliptical':
        amplitude = 0.1
        r_eff = 20
        n = 3
        theta = 0
        ellip = 0.5
        x_0, y_0 = 20, 0
    elif (gal_type == 'custom') | (gal_type == None):
        # Get args from kwargs
        amplitude = gal_params.get('amplitude', 0.1)
        r_eff = gal_params.get('r_eff', 20)
        n = gal_params.get('n', 1)
        theta = gal_params.get('theta', 0)
        ellip = gal_params.get('ellip', 0.5)
        x_0 = gal_params.get('x_0', 20)
        y_0 = gal_params.get('y_0', 0)
    
    mod = Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=x_0, y_0=y_0, ellip=ellip, theta=theta)
    gal = mod(x, y)
    
    return gal / ur.s

def construct_image(frame,pixel_size,exposure,psf_fwhm,read_noise,\
                    gal_type=None,gal_params=None,source=None,sky_rate=None):
    '''
        Return a simulated image with a background galaxy (optional), source (optional), and noise
        
        Required inputs:
        frame = Pixel dimensions of simulated image (30,30)
        pixel_size = Angular size of pixel (6 * ur.arcsec)
        exposure = Integration time of the frame (300 * ur.s)
        
        Optional inputs:
        galaxy = None OR default galaxy string ("spiral"/"elliptical") OR "custom" w/ galaxy params in kwargs
        source = None OR source photon count rate (100 / ur.s)
        
        To add:
        psf - make this an input itself, default to Gaussian w/ psf_fwhm. Work with oversampling
        num_exp - generate an array of exposures, all the same except different Poisson noise
    '''
    
    oversample = 6 # ~1 arcsec resolution for initial image generation
    
    # Make an oversampled PSF here
    pixel_size_init = pixel_size / oversample
    psf = gaussian_psf(psf_fwhm,(15,15),pixel_size_init)
    
    # Initialise an image, oversampled by the oversample parameter to begin with
    im_array = np.zeros(frame * oversample) / ur.s
    
    # Add a galaxy?
    if gal_type is not None:
        # Get a patch with a simulated galaxy on it
        gal = sim_galaxy(frame * oversample,pixel_size_init,gal_type=gal_type,gal_params=gal_params)
        im_array += gal
    
    # Add a source?
    if source is not None:
        # Place source as a delta function in the center of the frame
        im_array[im_array.shape[0] // 2 + 1, im_array.shape[1] // 2 + 1] += source
    
    # Convolve with the PSF (need to re-apply units here as it's lost in convolution)
    im_psf = convolve2d(im_array,psf,mode='same') / ur.s
    
    # Bin up the image by oversample parameter to the correct pixel size
    shape = (frame[0], oversample, frame[1], oversample)
    im_binned = im_psf.reshape(shape).sum(-1).sum(1)
    
    # Now add sky background
    if sky_rate is not None:
        # Add sky rate per pixel across the whole image
        im_binned += sky_rate
    
    # Convert to counts
    im_counts = im_binned * exposure

    # Apply Poisson noise and instrument noise
    im_noise = np.random.poisson(im_counts) + np.random.normal(loc=0,scale=read_noise,size=im_counts.shape)
    im_final = np.floor(im_noise)
    im_final[im_final < 0] = 0
    
    # Return image
    return im_final


def find(image,fwhm):
    '''
        Find all stars above the sky background level using DAOFind-like algorithm
    '''
    from photutils import Background2D
    from photutils.detection import DAOStarFinder
    
    # Create a background image
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (50, 50), filter_size=(5,5), bkg_estimator=bkg_estimator)
    sky = bkg.background_rms_median
    bkg_image = bkg.background
    print("Sky background rms: {}".format(sky))
    
    # Look for five-sigma detections
    threshold = 5 * sky
    
    # Find stars
    finder = DAOStarFinder(threshold,fwhm)
    star_tbl = finder.find_stars(image)
    print("Found {} stars".format(len(star_tbl)))
    
    return star_tbl, bkg_image, threshold

def run_daophot(image,psf,threshold,fwhm,star_tbl,niters):
    '''
        Given an image and a PSF, go run DAOPhot PSF-fitting algorithm
    '''
    from photutils.psf import DAOPhotPSFPhotometry
    
    # Fix star table columns
    star_tbl['x_0'] = star_tbl['xcentroid']
    star_tbl['y_0'] = star_tbl['ycentroid']
    
    # Initialise a Photometry object
    # This object loops find, fit and subtract
    photometry = DAOPhotPSFPhotometry(2.*fwhm,threshold,fwhm,psf,(3,3),niters=niters,aperture_radius=fwhm)
    
    # Problem with _recursive_lookup while fitting (needs latest version of astropy fix to modeling/utils.py)
    result = photometry(image=image, init_guesses=star_tbl)
    residual_image = photometry.get_residual_image()
    print("PSF-fitting complete")

    return result, residual_image

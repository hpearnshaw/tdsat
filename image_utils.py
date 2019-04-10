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
        gal_params = Dictionary of parameters for Sersic model: ...
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

def construct_image(frame,pixel_size,exposure,psf_fwhm,read_noise,oversample=6,psf=None,\
                    gal_type=None,gal_params=None,source=None,sky_rate=None,n_exp=1):
    '''
        Return a simulated image with a background galaxy (optional), source (optional), and noise
        
        Required inputs:
        frame = Pixel dimensions of simulated image (30,30)
        pixel_size = Angular size of pixel (6 * ur.arcsec)
        exposure = Integration time of the frame (300 * ur.s)
        psf_fwhm = FWHM in arcsec (4 * ur.arcsec)
        read_noise = Read noise in photons / pixel (unitless float)
        
        Optional inputs:
        oversample = How much to oversample image by when constructing. Must be consistent with psf.
        psf = 2D array containing PSF normalised to 1
        gal_type = Default galaxy string ("spiral"/"elliptical") or "custom" w/ Sersic parameters in gal_params
        gal_params = Dictionary of parameters for Sersic model (see sim_galaxy)
        source = Source photon count rate (100 / ur.s)
        n_exp = Number of exposures to be co-added
    '''
    pixel_size_init = pixel_size / oversample
    
    # Make an oversampled PSF if one is not already given
    if psf is None:
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

    # Apply jitter at this stage? Shuffle everything around by a pixel or something
    
    # Bin up the image by oversample parameter to the correct pixel size
    shape = (frame[0], oversample, frame[1], oversample)
    im_binned = im_psf.reshape(shape).sum(-1).sum(1)
    
    # Now add sky background
    if sky_rate is not None:
        # Add sky rate per pixel across the whole image
        im_binned += sky_rate
    
    # Convert to counts
    im_counts = im_binned * exposure

    # Co-add a number of separate exposures
    im_final = np.zeros(frame)
    for i in range(n_exp):
        # Apply Poisson noise and instrument noise
        im_noise = np.random.poisson(im_counts) + np.random.normal(loc=0,scale=read_noise,size=im_counts.shape)
        im_noise = np.floor(im_noise)
        im_noise[im_noise < 0] = 0

        # Add to the co-add
        im_final += im_noise
    
    # Return image 
    return im_final


def find(image,fwhm,method='daophot'):
    '''
        Find all stars above the sky background level using DAOFind-like algorithm
        
        Required inputs:
        image = 2D array of image on which to perform find
        fwhm = FWHM in pixels (1)
        
        Optional inputs:
        method = Either 'daophot' or 'peaks' to select different finding algorithms
    '''
    from photutils import Background2D, MedianBackground
    from photutils.detection import DAOStarFinder, find_peaks
    
    # Create a background image
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (image.shape[0] // 10, image.shape[1] // 10), \
                       filter_size=(3,3), bkg_estimator=bkg_estimator)
    sky = bkg.background_rms_median
    bkg_image = bkg.background
    print("Sky background rms: {}".format(sky))
    
    # Look for five-sigma detections
    threshold = 5 * sky
    
    # Find stars
    if method == 'daophot':
        finder = DAOStarFinder(threshold,fwhm)
        star_tbl = finder.find_stars(image)
        star_tbl['x'], star_tbl['y'] = star_tbl['xcentroid'], star_tbl['ycentroid']
    elif method == 'peaks':
        star_tbl = find_peaks(image-bkg_image,threshold,box_size=3)
        star_tbl['x'], star_tbl['y'] = star_tbl['x_peak'], star_tbl['y_peak']

    print("Found {} stars".format(len(star_tbl)))
    
    return star_tbl, bkg_image, threshold

def ap_phot(image,star_tbl,r=1.5,r_in=1.5,r_out=3.):
    '''
        Given an image, go do some aperture photometry
    '''
    from astropy.stats import sigma_clipped_stats
    from photutils import aperture_photometry, CircularAperture, CircularAnnulus

    # Build apertures from star_tbl
    positions = np.transpose([star_tbl['x'],star_tbl['y']])
    apertures = CircularAperture(positions, r=r)
    annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    annulus_masks = annulus_apertures.to_mask(method='center')

    # Get backgrounds in annuli
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(image)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)

    # Perform aperture photometry
    result = aperture_photometry(image, apertures)
    result['annulus_median'] = bkg_median
    result['aper_bkg'] = bkg_median * apertures.area()
    result['aper_sum_bkgsub'] = result['aperture_sum'] - result['aper_bkg']

    # To-do: get errors

    for col in result.colnames:
            result[col].info.format = '%.8g'  # for consistent table output
    print("Aperture photometry complete")

    return result, apertures, annulus_apertures

def run_daophot(image,threshold,fwhm,star_tbl,niters=1):
    '''
        Given an image and a PSF, go run DAOPhot PSF-fitting algorithm
    '''
    from photutils.psf import DAOPhotPSFPhotometry, IntegratedGaussianPRF
    
    # Fix star table columns
    star_tbl['x_0'] = star_tbl['x']
    star_tbl['y_0'] = star_tbl['y']
    
    # Define a fittable PSF model
    sigma = fwhm / (2. * np.sqrt(2 * np.log(2)))
    psf_model = IntegratedGaussianPRF(sigma=sigma)
    
    # Initialise a Photometry object
    # This object loops find, fit and subtract
    photometry = DAOPhotPSFPhotometry(3.*fwhm,threshold,fwhm,psf_model,(5,5),niters=niters,sigma_radius=5)
    
    # Problem with _recursive_lookup while fitting (needs latest version of astropy fix to modeling/utils.py)
    result = photometry(image=image, init_guesses=star_tbl)
    residual_image = photometry.get_residual_image()
    print("PSF-fitting complete")

    return result, residual_image

import galsim
import numpy as np
from astropy.table import Table

from bpp.catalog import validate_catalog
from bpp.galaxy import get_gaussian_galaxy_from_catalog


def create_gaussian_cutouts(
    slen: float,
    catalog: Table,
    psf: galsim.GSObject,
    pixel_scale: float = 0.2,
    g1: float = None,
    g2: float = None,
    sky_level: float = 0,
    seed: int = 0,
) -> np.ndarray:
    """Create cutouts of gaussian galaxies, one per row in catalog.

    Args:
        slen: Specify the side-length of the scene to produce.
        catalog: Astropy table with one entry per galaxy and its parameters.
        psf: Galsim object corresponding to PSF to use for convolving galaxies.
        g1: First reduced shear component to apply to all galaxies.
        g2: Second reduced shear component to apply to all galaxies.
        background: Background value to use.
        pixel_scale: Pixel scale to use [pixels / arcsecond]
        seed: To control randomness of noise added.

    Return:
        Numpy array with all cutouts of shape (n x slen x slen) where n
        is the number of rows in the catalog.
    """
    validate_catalog(catalog)
    cutouts = np.zeros((len(catalog), slen, slen))
    for ii, row in enumerate(catalog):
        galaxy = get_gaussian_galaxy_from_catalog(row)
        gal_conv = galsim.Convolve(galaxy, psf)
        gal_conv = gal_conv.shift(row["ra"], row["dec"])
        gal_conv.shear(g1=g1, g2=g2)
        image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale, bandpass=None)
        generator = galsim.random.BaseDeviate(seed=seed)
        noise = galsim.GaussianNoise(rng=generator, sigma=sky_level)
        image.addNoise(noise)
        cutouts[ii, :, :] = image.array
    return cutouts

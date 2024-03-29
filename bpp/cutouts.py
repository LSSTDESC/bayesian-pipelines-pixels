import galsim
import numpy as np

from bpp.catalog import validate_catalog
from bpp.galaxy import get_gaussian_galaxy_from_catalog


def create_gaussian_cutouts(
    slen: float,
    catalog: dict,
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
        catalog: Dictionary with numpy arrays corresponding to galaxy
            parameters, each row corresponds to a single cutout.
        psf: Galsim object corresponding to PSF to use for convolving galaxies.
        g1: First reduced shear component to apply to all galaxies.
        g2: Second reduced shear component to apply to all galaxies.
        background: Background value to use.
        pixel_scale: Pixel scale to use [pixels / arcsecond]
        sky_level: background sky level in counts.
        seed: To control randomness of noise added.

    Return:
        Numpy array with all cutouts of shape `(n x slen x slen)` where `n`
        is the number of rows in the catalog.
    """
    np.random.seed(seed)
    validate_catalog(catalog)
    n_rows = len(catalog["flux"])
    cutouts = np.zeros((n_rows, slen, slen))
    for i in range(n_rows):
        row = {key: catalog[key][i] for key in catalog}
        gal = get_gaussian_galaxy_from_catalog(row)
        gal = gal.shift(row["ra"], row["dec"])
        if g1 is not None and g2 is not None:
            gal = gal.shear(g1=g1, g2=g2)
        gal_conv = galsim.Convolve(gal, psf)
        img = gal_conv.drawImage(scale=pixel_scale, nx=slen, ny=slen, bandpass=None)
        img = img.array + sky_level
        img += np.random.randn(*img.shape) * np.sqrt(img)
        cutouts[i, :, :] = img
    return cutouts

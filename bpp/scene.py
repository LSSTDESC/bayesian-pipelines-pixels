from astropy.table import Table
import galsim
import numpy as np

from bpp.galaxy import get_bulge_disk_galaxy


def _validate_catalog(catalog: Table):
    """Ensure table is valid and has correct column names."""
    required = {'mag', 'a_d', 'a_b', 'b_b',
                'b_d', 'fluxnorm_d', 'beta', 'ra', 'dec'}
    if not required.issubset(set(catalog.colnames)):
        raise ValueError(f"Catalog does not have required columns: {required}")
    if not len(catalog) > 0:
        raise ValueError("Table has no row.")


def _convert_mag2flux(mag: float, exp_time: float, zeropoint: float):
    """Convert magnitude to flux."""
    return exp_time * 10 ** (-0.4 * (mag - zeropoint))


def create_scene(slen: float, catalog: Table, psf: galsim.GSObject,
                 pixel_scale: float = 0.2, g1: float = None, g2: float = None,
                 seed: int = 0) -> np.ndarray:
    """Create a scene of galaxies as desired positions specified in `catalog`.

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
        Numpy array containing an image with all galaxies of the catalog drawn.
    """
    _validate_catalog(catalog)
    gals = None
    for row in catalog:
        flux = _convert_mag2flux(row['mag'])
        fluxnorm_d, beta = row['fluxnorm_d'], row['beta']
        a_d, b_d, a_b, b_b = row['a_d'], row['b_d'], row['b_b'], row['b_d']
        galaxy = get_bulge_disk_galaxy(flux, fluxnorm_d,
                                       a_d, b_d, a_b, b_b, beta)
        gal_conv = galaxy.convolve(psf)
        gal_conv = gal_conv.shift(row["ra"], row["dec"])
        if gals is None:
            gals = gal_conv
        gals += gal_conv
    gals.shear(g1=g1, g2=g2)
    image = gals.drawImage(nx=slen, ny=slen, scale=pixel_scale)
    generator = galsim.random.BaseDeviate(seed=seed)
    noise = galsim.PoissonNoise(rng=generator, sky_level=0.0)
    image.add_noise(noise)
    return image.array

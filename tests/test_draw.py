from bpp.catalog import create_uniform_catalog
from bpp.psf import get_gaussian_psf
from bpp.scene import create_gaussian_cutouts


def test_draw_galaxy():
    """Test drawing scene."""
    cat = create_uniform_catalog(n_rows=5)
    psf = get_gaussian_psf(fwhm=0.7)
    cutouts = create_gaussian_cutouts(53, cat, psf, 0.2, 0, 0, 0, 0)
    assert cutouts.shape == (5, 53, 53)

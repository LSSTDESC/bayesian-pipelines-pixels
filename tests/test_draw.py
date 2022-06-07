from bpp.catalog import create_uniform_catalog
from bpp.psf import get_gaussian_psf
from bpp.scene import create_gaussian_scene


def test_draw_galaxy():
    """Test drawing scene."""
    cat = create_uniform_catalog()
    psf = get_gaussian_psf(fwhm=0.7)
    create_gaussian_scene(53, cat, psf, 0.2, 0, 0, 0, 0)

from bpp.catalog import create_uniform_catalog
from bpp.psf import get_gaussian_psf
from bpp.scene import create_scene


def test_draw_galaxy():
    cat = create_uniform_catalog()
    psf = get_gaussian_psf(fwhm=0.7)
    create_scene(slen=53, catalog=cat, psf=psf, pixel_scale=0.2,
                 g1=0.0, g2=0.0, sky_level=0.0, seed=0)

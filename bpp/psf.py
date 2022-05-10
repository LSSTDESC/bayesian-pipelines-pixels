import galsim
import numpy as np


def get_gaussian_psf(fwhm: float) -> galsim.GSObject:
    return galsim.Gaussian(fwhm=fwhm, flux=1.0)


def get_optical_and_atmospheric_psf(
    mirror_diameter: float,
    effective_area: float,
    filter_wavelength: float,
    atmospheric_fwhm: float,
    atmospheric_model: str = "Kolmogorov",
):
    """Defines a synthetic galsim PSF model.

    Credit: WeakLensingDeblending
        (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        mirror_diameter: in meters [m]
        effective_area: effective total light collecting area [m2]
        filter_wavelength: filter wavelength [Angstrom]
        fwhm: fwhm of the atmospheric component [arcsecond]
        atmospheric_model: type of atmospheric model. Options:
            ['Kolmogorov', 'Moffat']
    Returns:
        psf_model: galsim psf model
    """
    atmospheric_psf = None
    if atmospheric_model == "Kolmogorov":
        atmospheric_psf = galsim.Kolmogorov(fwhm=atmospheric_fwhm)
    elif atmospheric_model == "Moffat":
        atmospheric_psf = galsim.Moffat(2, fwhm=atmospheric_fwhm)
    else:
        raise NotImplementedError(
            f"The atmospheric model request '{atmospheric_model}'"
            f"is incorrect or not implemented."
        )

    optical_psf = None
    if mirror_diameter > 0:
        mirror_area = np.pi * (0.5 * mirror_diameter) ** 2
        area_ratio = effective_area / mirror_area
        if area_ratio <= 0 or area_ratio > 1:
            msg = "Incompatible effective-area and mirror-diameter values."
            raise RuntimeError(msg)
        obscuration_fraction = np.sqrt(1 - area_ratio)
        lambda_over_diameter = 1e-10 * filter_wavelength / mirror_diameter
        lambda_over_diameter = 3600 * np.degrees(lambda_over_diameter)
        optical_psf = galsim.Airy(
            lam_over_diam=lambda_over_diameter,
            obscuration=obscuration_fraction
        )

    # define the psf model according to the components we have
    use_atmos = isinstance(atmospheric_psf, galsim.GSObject)
    use_optic = isinstance(optical_psf, galsim.GSObject)
    if use_atmos and use_optic:
        psf = galsim.Convolve(atmospheric_psf, optical_psf)
    elif use_atmos and not use_optic:
        psf = atmospheric_psf
    elif use_optic and not use_atmos:
        psf = optical_psf
    else:
        raise RuntimeError("Neither component is defined.")
    return psf.withFlux(1.0)

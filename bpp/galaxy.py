import galsim
import numpy as np


def get_gaussian_galaxy(
    flux: float, hlr: float, q: float, beta: float
) -> galsim.GSObject:
    """Returns a single Gaussian galaxy galsim object."""
    beta_radians = beta * galsim.radians
    circular_gal = galsim.Gaussian(flux=flux, half_light_radius=hlr)
    return circular_gal.shear(q=q, beta=beta_radians)


def get_gaussian_galaxy_from_catalog(row: dict) -> galsim.GSObject:
    """Returns the corresponding gaussian galaxy for a given row in the catalog."""
    # NOTE: We take the HLR to be one from the disk component in catalog.
    flux = row["flux"].item()
    hlr = np.sqrt(row["a_d"] * row["b_d"]).item()
    q = row["b_d"].item() / row["a_d"].item()
    beta = row["beta"].item()
    return get_gaussian_galaxy(flux, hlr, q, beta)


def get_sersic_galaxy(
    n: float, flux: float, hlr: float, q: float, beta: float
) -> galsim.GSObject:
    """Returns a single Sersic galaxy galsim object."""
    beta_radians = beta * galsim.radians
    circular_gal = galsim.Sersic(n=n, flux=flux, half_light_radius=hlr)
    return circular_gal.shear(q=q, beta=beta_radians)


def get_sersic_galaxy_from_catalog(row: dict) -> galsim.GSObject:
    """Returns the corresponding Sersic galaxy for a given row in the catalog."""
    # NOTE: We take the HLR to be one from the disk component in catalog.
    flux = row["flux"].item()
    hlr = np.sqrt(row["a_d"] * row["b_d"]).item()
    n = row["n"]
    q = row["b_d"].item() / row["a_d"].item()
    beta = row["beta"].item()
    return get_sersic_galaxy(n, flux, hlr, q, beta)


def get_bulge_disk_galaxy(
    flux: float,
    fluxnorm_d: float,
    a_d: float,
    b_d: float,
    a_b: float,
    b_b: float,
    beta: float,
) -> galsim.GSObject:
    """Returns a Galsim model of a combined Bulge+Disk Galaxy.

    Both the bulge and disk component are assume to have the same orientation
    specified by `beta`.

    Args:
        flux: Total Flux of disk + bulge model (counts).
        fluxnorm_d: Fraction of flux belong to disk. It is assumed that
            `1 - fluxnorm_d` is the fraction belonging to the bulge.
        a_d: Semi-major axis of disk component (arcsecs).
        b_d: Semi-minor axis of disk component (arcsecs).
        a_b: Semi-major axis of bulge component (arcsecs).
        b_b: Semi-minor axis of bulge component (arcsecs).
        beta: Orientation of semi-major axis w.r.t horizontal axis (radians).

    Returns:
        Galsim Object consisting of a Bulge and Disk component.
    """
    flux_d = flux * fluxnorm_d
    flux_b = flux * (1 - fluxnorm_d)
    beta_radians = beta * galsim.radians  # same for both bulge+disk.

    if flux_d + flux_b == 0:
        raise ValueError("Source not visible")

    components = []

    if flux_d > 0:
        disk_hlr = np.sqrt(a_d * b_d)
        disk_q = b_d / a_d
        disk = galsim.Exponential(flux=flux_d, half_light_radius=disk_hlr)
        disk = disk.shear(q=disk_q, beta=beta_radians)
        components.append(disk)

    if flux_b > 0:
        bulge_hlr = np.sqrt(a_b * b_b)
        bulge_q = b_b / a_b
        bulge = galsim.DeVaucouleurs(flux=flux_b, half_light_radius=bulge_hlr)
        bulge = bulge.shear(q=bulge_q, beta=beta_radians)
        components.append(bulge)

    return galsim.Add(components)


def get_bulge_disk_galaxy_from_catalog(row: dict) -> galsim.GSObject:
    """Returns the corresponding Bulge+Disk galaxy for a given row in the catalog."""
    flux, fluxnorm_d = row["flux"].item(), row["fluxnorm_d"].item()
    a_d, a_b = row["a_d"].item(), row["a_b"].item()
    b_b, b_d = row["b_b"].item(), row["b_d"].item()
    beta = row["beta"].item()
    return get_bulge_disk_galaxy(flux, fluxnorm_d, a_d, b_d, a_b, b_b, beta)

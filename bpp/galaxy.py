import galsim
import numpy as np


def get_bulge_disk_galaxy(flux: float, fluxnorm_d: float,
                          a_d: float, b_d: float, a_b: float, b_b: float,
                          beta: float) -> galsim.GSObject:
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
    beta_radians = beta * galsim.radians

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
        disk_q = b_d / a_d
        disk = galsim.Exponential(flux=flux_d, half_light_radius=disk_hlr)
        disk = disk.shear(q=disk_q, beta=beta_radians)
        bulge = galsim.DeVaucouleurs(flux=flux_b, half_light_radius=bulge_hlr)
        bulge = bulge.shear(q=disk_q, beta=beta_radians)
        components.add(bulge)

    return galsim.add(components)

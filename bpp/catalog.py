import numpy as np
from astropy.table import Table


def validate_catalog(catalog: Table):
    """Ensure table is valid and has correct column names."""
    required = {"flux", "fluxnorm_d", "a_d", "a_b", "b_b", "b_d", "beta", "ra", "dec"}
    if not required.issubset(set(catalog.colnames)):
        raise ValueError(f"Catalog does not have required columns: {required}")
    if not len(catalog) > 0:
        raise ValueError("Catalog has no rows.")


def create_uniform_catalog(
    n_rows: int = 1,
    max_shift: float = 4.0,
    min_a: float = 0.7,
    max_a: float = 5.0,
    min_q: float = 0.5,
    n_sersic_bins: int = 10,
):
    """Creates a galaxy catalog with uniform parameters.

    Args:
        n_rows: How many rows in the catalog?
        max_shift: Maximum ra,dec shift in arcseconds.
        min_a: Minimum semi-major axis for both bulge and disk component.
        max_a: Maximum semi-major axis for both bulge and disk component.
        n_sersic_bins: Number of bins up to 100 that represent number of possible
            sersic indices sampled between 1 and 4.

    Returns:
        Astropy table with rows corresponding to a single galaxy.
    """
    assert n_sersic_bins <= 100, "See Galsim documentation on Sersics."
    assert max_shift / 0.2 <= 10, "Limit to single uncentered galaxies on a cutout."
    n_bins = np.linspace(1, 4, n_sersic_bins)  # galsim does not like continuous index.
    n = np.random.choice(n_bins, size=n_rows, replace=True)
    flux = 10 ** np.random.uniform(3, 6, size=n_rows)
    fluxnorm_d = np.random.uniform(size=n_rows)
    beta = np.random.uniform(0, 2 * np.pi, size=n_rows)
    a_d = np.random.uniform(min_a, max_a, size=n_rows)
    a_b = np.random.uniform(min_a, max_a, size=n_rows)
    q_d = np.random.uniform(min_q, 1, size=n_rows)
    q_b = np.random.uniform(min_q, 1, size=n_rows)
    b_d = a_d * q_d
    b_b = a_b * q_b
    ra = np.random.uniform(-max_shift, max_shift, size=n_rows)
    dec = np.random.uniform(-max_shift, max_shift, size=n_rows)
    data = [flux, fluxnorm_d, beta, a_d, a_b, b_d, b_b, n, ra, dec]
    names = ["flux", "fluxnorm_d", "beta", "a_d", "a_b", "b_d", "b_b", "n", "ra", "dec"]
    return Table(data=data, names=names)

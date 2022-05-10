import numpy as np
from astropy.table import Table


def create_uniform_catalog(n_rows: int = 1, max_shift: float = 4.0,
                           min_a: float = 0.7, max_a: float = 5.0):
    """Creates a galaxy catalog with uniform parameters.

    Args:
        n_rows: How many rows in the catalog?
        max_shift: Maximum ra,dec shift in arcseconds.
        min_a: Minimum semi-major axis for both bulge and disk component.
        max_a: Maximum semi-major axis for both bulge and disk component.

    Returns:
        Astropy table with rows corresponding to a single galaxy.
    """
    flux = 10**np.random.uniform(3, 6, size=n_rows)
    fluxnorm_d = np.random.uniform(size=n_rows)
    beta = np.random.uniform(0, 2 * np.pi, size=n_rows)
    a_d = np.random.uniform(min_a, max_a, size=n_rows)
    a_b = np.random.uniform(min_a, max_a, size=n_rows)
    q_d = np.random.uniform(0, 1, size=n_rows)
    q_b = np.random.uniform(0, 1, size=n_rows)
    b_d = a_d * q_d
    b_b = a_b * q_b
    ra = np.random.uniform(-max_shift, max_shift, size=n_rows)
    dec = np.random.uniform(-max_shift, max_shift, size=n_rows)
    data = [flux, fluxnorm_d, beta, a_d, a_b, b_d, b_b, ra, dec]
    names = ['flux', 'fluxnorm_d', 'beta',
             'a_d', 'a_b', 'b_d', 'b_b', 'ra', 'dec']
    return Table(data=data, names=names)

import galcheat
import numpy as np
from galcheat.utilities import mag2counts


def validate_catalog(catalog: dict):
    """Ensure table is valid and has correct column names."""
    required = {"flux", "fluxnorm_d", "a_d", "a_b", "b_b", "b_d", "beta", "ra", "dec"}
    n_rows = len(catalog["flux"])
    for key, value in catalog.items():
        assert value.shape == (n_rows,), f"Column {key} has wrong shape."
    if not required.issubset(set(catalog.keys())):
        raise ValueError(f"Catalog does not have required columns: {required}")
    if not n_rows > 0:
        raise ValueError("Catalog has no rows.")


def create_uniform_catalog(
    n_rows: int = 1,
    max_shift: float = 1.0,
    min_a: float = 0.7,
    max_a: float = 3.0,
    min_q: float = 0.5,
    n_sersic_bins: int = 10,
    survey_name: str = "LSST",
    filter_name: str = "r",
):
    """Creates a galaxy catalog with uniform parameters.

    Args:
        n_rows: How many rows in the catalog?
        max_shift: Maximum ra,dec shift in arcseconds.
        min_a: Minimum semi-major axis for both bulge and disk component, in
            arcseconds.
        max_a: Maximum semi-major axis for both bulge and disk component, in
            arcseconds.
        n_sersic_bins: Number of bins up to 100 that represent number of
            possible sersic indices sampled between 1 and 4.
        survey_name: Name of survey, must be supported by Galcheat.
        filter_name: Name of filter, must be supported by Galcheat.

    Returns:
        A dictionary containing the catalog.
    """
    assert n_sersic_bins <= 100, "See Galsim documentation on Sersics."
    assert max_shift / 0.2 <= 5, "Limit to single uncentered galaxies on a cutout."

    # first we sample magnitude and get flux from it.
    survey = galcheat.get_survey(survey_name)
    filter_ = survey.get_filter(filter_name)
    mag = np.random.uniform(18, 25.3, size=n_rows)  # from btk sample catalog
    flux = mag2counts(mag, survey, filter_).to_value("electron")

    # now we sample all other parameters.
    # galsim slows with continuous index, so we sample n_sersic in bins.
    n_bins = np.linspace(1, 4, n_sersic_bins)
    n = np.random.choice(n_bins, size=n_rows, replace=True)
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
    data = [mag, flux, fluxnorm_d, beta, a_d, a_b, b_d, b_b, n, ra, dec]
    names = ["mag", "flux", "fluxnorm_d", "beta", "a_d"]
    names += ["a_b", "b_d", "b_b", "n", "ra", "dec"]
    return dict(zip(names, data))

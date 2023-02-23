from configparser import ConfigParser

from math import log
from scipy.stats import gamma as gamma_distribution
# from scipy.stats import invgamma as gamma_distribution


class GalaxyDistPrior():

    def __init__(self):
        self._ngals = 0

    def set_ngals(self, ngals):

        self._ngals = ngals


class GalaxyDistPriorConst(GalaxyDistPrior):

    def __init__(self, a=3.0, b=0.5):
        super().__init__()

        self._sigma_e_sq = 0.09
        self._a = a
        self._b = b
        self._beta = 0.
        self._n = 0

    def get_lnjac_omega(self, igal):

        return 1.

    def init_output_data(self):
        pass

    def set_parameters(self, ini: ConfigParser, verbose):
        """_summary_

        Parameters
        ----------
        ini : configparser.ConfigParser
            _description_
        verbose : _type_
            _description_
        """

        self._sigma_e_sq = ini.getfloat(
            "const_ellip_dist", "sigma_e_sq", fallback=0.09
        )
        self._a = ini.getfloat(
            "const_ellip_dist", "a_sigma_sq_prior", fallback=3.0
        )
        self._b = ini.getfloat(
            "const_ellip_dist", "b_sigma_sq_prior", fallback=0.5
        )

    def get_params(self, igal):
        return self._sigma_e_sq

    def ngroups(self):
        return 1

    def n_in_group(self, ndx):
        return self._ngals

    def gal_index_from_group(self, i, ndx):
        return i

    def ln_prior(self):
        return -(self._a-1)*log(self._sigma_e_sq) - self._b/self._sigma_e_sq

    def r_over_b(self, igal, p):
        return 0.

    def update_latent_param(
        self, i, p, lnp_gal, lnp_marg, rng_eng
    ):
        pass

    def reset_interim_sample_aggregation(self):
        self._beta = 0.
        self._n = 0

    def update_group_params(self, ndx, rng_eng):
        """

        NOTE: The scipy distribution use a different convention than the C
        version so we have scale=beta for scipy instead of 1/beta

        Parameters
        ----------
        ndx : _type_
            _description_
        rng_eng : _type_
            numpy random generator np.random.Generator(np.random.MT19937(seed))
        """
        beta = self._b + self._beta
        gamma = gamma_distribution
        gamma.random_state = rng_eng
        x = gamma.rvs(self._n + self._a, scale=1./beta)
        self._sigma_e_sq = 1. / x

    def get_aux_param(self, igal):
        return 0.

from configparser import ConfigParser

from math import log
from scipy.stats import uniform as uniform_distribution
from scipy.stats import norm as normal_distribution
import numpy


CURRENT = 0
PROPOSAL = 1


class ShearModel():

    def __init__(self):

        self._shear_acceptance_count = 0

    def reset_shear_acceptance_count(self):
        self._shear_acceptance_count = 0

    def get_shear_acceptance_count(self):
        return self._shear_acceptance_count


class ShearModelConst(ShearModel):

    def __init__(self):
        super().__init__()

        self._g_proposal_dist = ConstShearProposal()

        self._g1 = 0.
        self._g2 = 0.
        self._kappa = 0.
        self._g1_prop = 0.
        self._g2_prop = 0.
        self._kappa_prop = 0.
        self._prior_var = 1.

    def InitOutputData(self, nsteps):
        self._shear_out = {
            "g1": [-10]*nsteps,
            "g2": [-10]*nsteps,
            "kappa": [-10]*nsteps,
        }

        self._lnp_out = {"lnp": [-999999]*nsteps}

    def SetParameters(self, ini: ConfigParser):

        self._prior_var = ini.getfloat(
            "const_shear", "prior_var", fallback=2.0
        )
        prop_sd = ini.getfloat(
            "const_shear", "shear_prop_sd", fallback=5e-3
        )
        self._g_proposal_dist.set_prop_sd(prop_sd)

    def get_params(self, igal, selector):

        res = [0]*3
        if selector == CURRENT:
            res[0] = self._g1
            res[1] = self._g2
            res[2] = self._kappa
        elif selector == PROPOSAL:
            res[0] = self._g1_prop
            res[1] = self._g2_prop
            res[2] = self._kappa_prop
        else:
            raise ValueError(
                f"ShearModelConst - Bad value for selector: {selector}"
            )

        return res

    def GetReducedShear(self, igal, selector):
        """
        The parameters are already reduced shear - no transformation should be
        done

        Parameters
        ----------
        igal : _type_
            _description_
        selector : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.get_params(igal, selector)

    def set_params(self, p, selector):

        if selector == CURRENT:
            self._g1 = p[0]
            self._g2 = p[1]
            self._kappa = p[2]
        elif selector == PROPOSAL:
            self._g1_prop = p[0]
            self._g2_prop = p[1]
            self._kappa_prop = p[2]
        else:
            raise ValueError(
                f"ShearModelConst - Bad value for selector: {selector}"
            )

    def set_prior_var(self, var):
        self._prior_var = var

    def LnPior(self, selector):

        p = [0]*3
        p = self.get_params(0, selector)
        chisq = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]) / self._prior_var
        return -0.5*chisq

    def propose(self):

        g = self.get_params(0, CURRENT)
        g_new = self._g_proposal_dist.Propose(g)
        self.set_params(g_new, PROPOSAL)

    def accept(self):

        self._g1 = self._g1_prop
        self._g2 = self._g2_prop
        self._kappa = self._kappa_prop

    def update(self, lnp, lnp1):

        unif = uniform_distribution

        if self._g_proposal_dist.IsGood():
            h = min(0., lnp1-lnp)
            unif.random_state = self._g_proposal_dist._rng_eng
            u = unif.rvs()
            if log(u) <= h:
                self.accept()
                lnp = lnp1
                self._shear_acceptance_count += 1
        return lnp

    def save_step(self, istep, lnp):

        self._shear_out["g1"][istep] = self._g1
        self._shear_out["g2"][istep] = self._g2
        self._shear_out["kappa"][istep] = self._kappa

        self._lnp_out["lnp"][istep] = lnp


class ConstShearProposal():
    """
    Proposal distribution for constant shear MH updates
    """

    def __init__(
            self,
            rng_eng=numpy.random.Generator(numpy.random.MT19937(42))
    ):

        self._is_good = False
        self._shear_prop_sd = 5.0e-3

        # This need to be set somehow from Thresher
        self._rng_eng = rng_eng

    def set_prop_sd(self, sd):
        self._shear_prop_sd = sd

    def Propose(self, g):

        rnorm1 = normal_distribution
        rnorm1.random_state = self._rng_eng
        rnorm2 = normal_distribution
        rnorm2.random_state = self._rng_eng
        rnorm3 = normal_distribution
        rnorm3.random_state = self._rng_eng

        x = rnorm1.rvs(0., 1.)
        y = rnorm2.rvs(0., 1.)
        z = rnorm3.rvs(0., 1.)

        res = [0]*3
        res[0] = g[0] + self._shear_prop_sd * x
        res[1] = g[1] + self._shear_prop_sd * y
        res[2] = g[2] + self._shear_prop_sd * z
        self._is_good = True

        return res

    def IsGood(self):
        return self._is_good

from math import log, exp


LOGTWOPI = 1.8378770664093453


def unshear(e1, e2, g1, g2, weak_shear):
    """
    Subtract the reduced shear g from the ellipticity e.

    Parameters
    ----------
    e1 : _type_
        _description_
    e2 : _type_
        _description_
    g1 : _type_
        _description_
    g2 : _type_
        _description_
    weak_shear : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    d1 = e1 - g1
    d2 = e2 - g2
    if weak_shear:
        e1_int = d1
        e2_int = d2
    else:
        f1 = 1. - e1*g1 - e2*g2
        f2 = e2*g1 - e1*g2
        denom = f1*f1 + f2*f2

        e1_int = (d1*f1 - d2*f2) / denom
        e2_int = (d2*f1 - d1*f2) / denom

    return e1_int, e2_int


class GalaxyModelPrior():
    """
    A Base class for the prior on galaxy model parameters
    """

    def __init__(self, g1=0., g2=0., kappa=0., weak_shear=False):

        self._N_GAL_PARAMS = 4

        self._g1 = g1
        self._g2 = g2
        self._weak_shear = weak_shear
        if weak_shear:
            self._kappa = 0.
        else:
            self._kappa = kappa
        self.set_magnification()
        self._lognorm = 0.

    def set_magnification(self):

        if self._weak_shear:
            gamma1 = self._g1
            gamma2 = self._g2
        else:
            # g1_, g2_ are reduced shear, so we must convert to non-reduced
            # shear
            gamma1 = (1 - self._kappa) * self._g1
            gamma2 = (1 - self._kappa) * self._g2

        self._magnification = 1.0 / \
            (
                (1. - self._kappa) * (1. - self._kappa)
                - (gamma1*gamma1 + gamma2*gamma2)
             )

    def get_unlensed_params(self, e1, e2, ln_r_e, ln_F):

        e_int = unshear(e1, e2, self._g1, self._g2, self._weak_shear)
        e1 = e_int[0]
        e2 = e_int[1]
        ln_mu = log(self._magnification)
        ln_r_e = ln_r_e - ln_mu
        ln_F = ln_F - ln_mu

        return e1, e2, ln_r_e, ln_F

    def set_lensing_params(self, p):

        self._g1 = p[0]
        self._g2 = p[1]
        if self._weak_shear:
            self._kappa = 0.
        else:
            self._kappa = p[2]
        self.set_magnification()


class EllipticityPriorGaussian(GalaxyModelPrior):
    """
    A Gaussian functional form for the final prior of e1, e2

    Parameters
    ----------
    GalaxyModelPrior : _type_
        _description_
    """

    def __init__(self, g1=0, g2=0, ln_sigma_e=0, weak_shear=False):
        super().__init__(g1, g2, 0., weak_shear)

        self._ln_sigma_e = ln_sigma_e

        self._sigma_e_sq = exp(2. * ln_sigma_e)
        self._lognorm = -LOGTWOPI - 2. * ln_sigma_e

    def set_hyperprior_params(self, p):

        self._ln_sigma_e = log(p) / 2.
        self._sigma_e_sq = p
        self._lognorm = -LOGTWOPI - 2.*self._ln_sigma_e

    def eval(self, e1, e2):

        ln_r_e = 0.
        ln_F = 0.
        e1, e2, ln_r_e, ln_F = self.get_unlensed_params(
            e1, e2, ln_r_e, ln_F
        )

        e_sq = e1*e1 + e2*e2

        if e_sq < 0.:
            return -73.6
        else:
            return -0.5 * (e_sq / self._sigma_e_sq) + self._lognorm

    def get_gradiant(self, e1, e2):

        ln_r_e = 0.
        ln_F = 0.

        e1, e2, _, _ = self.get_unlensed_params(e1, e2, ln_r_e, ln_F)

        if self._weak_shear:
            de1_dgamma1 = -1.
            de1_dgamma2 = 0.
            de1_dkappa = 0.

            de2_dgamma1 = 0.
            de2_dgamma2 = -1.
            de2_dkappa = 0.
        else:
            raise ValueError(
                "Ellipticity prior gradient not implemented for non-weak shear"
            )

        e_sq = e1*e1 + e2*e2

        de_sq = [0]*3
        de_sq[0] = e1 * de1_dgamma1 + e2 * de2_dgamma1
        de_sq[1] = e1 * de1_dgamma2 + e2 * de2_dgamma2
        de_sq[2] = e1 * de1_dkappa + e2 * de2_dkappa

        pr = exp(-0.5*(e_sq / self._sigma_e_sq) - self._lognorm)

        res = []
        for i in range(0, 3):
            res.append(-pr * de_sq[i] / self._sigma_e_sq)

        return res

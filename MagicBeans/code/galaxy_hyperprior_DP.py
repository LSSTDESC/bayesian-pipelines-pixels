from configparser import ConfigParser

from galaxy_hyperprior import GalaxyDistPrior
from latent_classes import LatentClasses
from dp_params import DPParams

from math import log
from scipy.stats import gamma as gamma_distribution
from scipy.stats import uniform as uniform_distribution
from scipy.stats import norm as normal_distribution


def Pr_Marg(e1, e2, a, b):
    """
    Prior for e_int marginalized over sigma_e_sq with prior G0

    Parameters
    ----------
    e1 : _type_
        _description_
    e2 : _type_
        _description_
    a : _type_
        _description_
    b : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    e_int_sq = e1*e1 + e2*e2

    beta = 1 + e_int_sq/(2*b)
    coef = a / (b*6.283185307179586232)

    return coef * beta**(-(1 + a))


class GalaxyDistPriorDP(GalaxyDistPrior):
    """
    DP-distributed alpha

    Parameters
    ----------
    GalaxyDistPrior : _type_
        _description_
    """

    def __init__(self, a=3., b=0.5, dp_param_val=0.09):
        super().__init__()

        self._latent_classes = LatentClasses()
        self._dp_params = DPParams()
        self._dp_params.init_params_values(dp_param_val)

        self._a = a
        self._b = b
        self._dp_precision = 0.02
        self._scale_param_prec = 1000.
        self._scale_param_mean = 1.
        self._dp_precision_prior_a = 0.547
        self._dp_precision_prior_b = 0.107
        self._sample_dp_precision = True

        # The mode of the Gamma distribution prior is
        # (a_tau_prior - 1) /b_tau_prior

        # 'shape' parameter for the Gamma distribution
        self._a_tau_prior = 1001.
        # 'rate' parameter for the Gamma distribution
        self._b_tau_prior = 10.

        self._m_mean_prior = 1.
        self._m_var_prior = 0.01

        self._sample_scale_param_mean = False

        self._n_lc_max = 100
        self._save_lc_labels = True

    def gat_lnjac_omega(self, igal):
        return abs(self._scale_params[igal])

    def Allocate(self):
        self._latent_classes.Allocate(self._ngals)
        self._scale_params = [1. for i in range(self._ngals)]
        self._beta = [0. for i in range(self._n_lc_max)]

    def SetParameters(self, ini: ConfigParser, verbose=True):
        """_summary_

        Parameters
        ----------
        ini : configparser.ConfigParser
            _description_
        verbose : bool, optional
            _description_, by default True
        """

        if verbose:
            print("<GalaxyDistPriorDP> Setting parameters from INI file")
        dp_param_val = ini.getfloat(
            "dp", "dp_parameter_value", fallback=0.056
        )
        self._dp_params.init_params_values(dp_param_val)

        self._dp_precision = ini.getfloat(
            "dp", "dp_precision", fallback=0.01
        )
        self._n_lc_max = ini.getint(
            "dp", "n_lc_max", fallback=20
        )
        self._save_lc_labels = ini.getboolean(
            "dp", "save_lc_labels", fallback=True
        )

        self._dp_precision_prior_a = ini.getfloat(
            "dp", "dp_precision_prior_a", fallback=0.547
        )
        self._dp_precision_prior_b = ini.getfloat(
            "dp", "dp_precision_prior_b", fallback=0.107
        )
        self._sample_dp_precision = ini.getboolean(
            "sampling", "sample_dp_precision", fallback=True
        )
        self._a = ini.getfloat("dp", "a_sigma_sq_prior", fallback=3.0)
        self._b = ini.getfloat("dp", "b_sigma_sq_prior", fallback=0.5)

        self._a_tau_prior = ini.getfloat(
            "scale_params", "a_tau_prior", fallback=1001.0
        )
        self._b_tau_prior = ini.getfloat(
            "scale_params", "b_tau_prior", fallback=10.0
        )
        self._m_mean_prior = ini.getfloat(
            "scale_params", "m_mean_prior", fallback=1.0
        )
        self._m_var_prior = ini.getfloat(
            "scale_params", "m_var_prior", fallback=0.01
        )
        self._sample_scale_param_mean = ini.getboolean(
            "sampling", "sample_scale_param_mean", fallback=False
        )

        if verbose:
            print("--- <GalaxyDistPriorDP>Parameters set from INI file:")
            print("\tdp_param_val:", dp_param_val)
            print("\tdp_precision:", self._dp_precision)
            print("\tn_lc_max:", self._n_lc_max)
            print("\tdp_precision_prior_a:", self._dp_precision_prior_a)
            print("\tdp_precision_prior_b:", self._dp_precision_prior_b)
            print("\ta_sigma_sq_prior:", self._a)
            print("\tb_sigma_sq_prior:", self._b)
            # print("\ta_tau_prior: %5.4f\n", a_tau_prior_)
            # print("\tb_tau_prior: %5.4f\n", b_tau_prior_)
            # print("\tm_mean_prior: %5.4f\n", m_mean_prior_)
            # print("\tm_var_prior: %5.4f\n", m_var_prior_)
            print("\tsample_scale_param_mean:", self._sample_scale_param_mean)

    def HDraw(self, i, p, rng_eng):
        """_summary_

        Draw from the posterior distribution for alpha with prior G0 and
        observation i.

        Given only the samples for e_int, first draw e_int ~ p(e_int | y_i) and
        then draw alpha ~ p(alpha | e_int) to get joint samples for e_int and
        alpha.
        By discarding the e_int values we get samples from the marginal
        posterior for alpha.

        Parameters
        ----------
        i : _type_
            _description_
        p : _type_
            _description_
        rng_eng : _type_
            numpy random generator np.random.Generator(np.random.MT1993(seed))

        Returns
        -------
        _type_
            _description_
        """

        beta = self._b + 0.5 * (p[0]*p[0] + p[1]*p[1])
        gamma = gamma_distribution
        gamma.random_state = rng_eng
        x = gamma.rvs(1 + self._a, scale=1./beta)
        return 1. / x

    def HDrawCluster(self, latent_class_val, rng_eng):
        """
        Draw new value of sigma_eta_sq associated with latent class
        latent_class_val

        Parameters
        ----------
        latent_class_val : _type_
            _description_
        rng_eng : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        n_obs_clust = self._latent_classes.GetNumGalInCluster(
            latent_class_val
        )
        beta = self._b + self._beta[latent_class_val]
        gamma = gamma_distribution
        gamma.random_state = rng_eng
        x = gamma.rvs(n_obs_clust + self._a, scale=1./beta)
        return 1. / x

    def r_over_b(self, igal, p):
        """
        Marginal likelihood of galaxy i

        Parameters
        ----------
        igal : _type_
            _description_
        p : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        a = 1 + self._a
        b = self._b
        xi = abs(self._scale_params[igal])
        jac = xi
        eta1 = p[0] / xi
        eta2 = p[1] / xi
        r = jac * Pr_Marg(eta1, eta2, a, b)
        return r

    def DrawLatentClassAssignement(self, p, np, rng_eng):
        """
        Draw an integer on the interval [0, np-1] with probabilities p
        param p list of probabilities that sum to 1

        Parameters
        ----------
        p : _type_
            _description_
        np : _type_
            _description_
        rng_eng : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        unif = uniform_distribution
        unif.random_state = rng_eng
        u = unif.rvs()

        p_cdf = 0
        for latent_class in range(np):
            p_cdf += p[latent_class]
            if u <= p_cdf:
                break

        return latent_class

    def GibbsUpdateDPPrecision(self, rng_eng):
        """
        ! Draw the DP precision parameter from the conditional posterior.

        Reference
        ---------
        [1] M. D. Escobar and M. West, "Bayesian Density Estimation and
        Inference Using Mixtures,"
        Journal of the American Statistical Association 90(430), 577-588(1995)
        [doi:10.1080/01621459.1995.10476550].
        Eqns. (13-14) and following paragraph.


        Parameters
        ----------
        rng_eng : _type_
            _description_
        """

        n = self._ngals
        a = self._dp_precision_prior_a
        b = self._dp_precision_prior_b
        k = self.ngroups()

        # Draw eta from a Beta distribution by means of 2 gamma-distributed
        # RVs.
        # http://stackoverflow.com/a/10359049/4907
        gamma1 = gamma_distribution
        gamma1.random_state = rng_eng
        gamma2 = gamma_distribution
        gamma2.random_state = rng_eng

        X = gamma1.rvs(self._dp_precision + 1., scale=1.)
        Y = gamma2.rvs(n, scale=1.)

        eta = X / (X + Y)
        ln_eta = log(eta)

        pi_eta = (a + k - 1) / (a + k - 1 + n * (b - ln_eta))

        unif = uniform_distribution
        unif.random_state = rng_eng
        u = unif.rvs()
        if u <= pi_eta:
            alpha = a + k
        else:
            alpha = a + k - 1

        beta = b - ln_eta

        gammadist = gamma_distribution
        gammadist.random_state = rng_eng
        self._dp_precision = gammadist.rvs(alpha, scale=1./beta)

    def ScalingParamLnPosterior(self, xi_i, lnp_gal):

        delta_xi = xi_i - self._scale_param_mean
        return lnp_gal + (-0.5 * delta_xi * delta_xi * self._scale_param_prec)

    def propose_aux_param(self, igal, rng_eng):
        """_summary_

        Note never called

        Parameters
        ----------
        igal : _type_
            _description_
        rng_eng : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        sigma_xi = 0.25
        xi = self._scale_params[igal]
        rnorm = normal_distribution
        rnorm.random_state = rng_eng
        xi_new = rnorm.rvs(xi, sigma_xi)
        self._scale_params[igal] = xi_new

        return xi_new

    def get_params(self, igal):

        xi = self._scale_params[igal]
        return self._dp_params[self._latent_classes[igal]] * xi * xi

    def get_params_for_group(self, igal, igroup):

        xi = self._scale_params[igal]
        return self._dp_params[igroup] * xi * xi

    def get_aux_params(self, igal):
        return self._scale_params[igal]

    def ngroups(self):
        return self._latent_classes.GetNumClusters()

    def n_in_group(self, ndx):
        return self._latent_classes.GetNumGalInCluster(ndx)

    def gal_index_from_group(self, i, ndx):
        igal = [0]*self.n_in_group(ndx)
        igal = self._latent_classes.GetGalIndicesInCluster(ndx, igal)
        return igal[i]

    def LnPrior(self):
        return 0.

    def reset_interim_sample_aggregation(self):
        for i in range(self._n_lc_max):
            self._beta[i] = 0

    def aggregate_interim_samples(self, igal, ndx, p):

        assert igal < self._ngals
        # beta_ is only allocated for n_lc_max_ classes
        assert ndx < self._n_lc_max
        xi = abs(self._scale_params[igal])
        e1 = p[0] / xi
        e2 = p[1] / xi
        self._beta[ndx] += 0.5 * (e1*e1 + e2*e2)

    def update_group_params(self, ndx, rng_eng):

        lc = self._latent_classes.UniqueClusters(ndx)
        new_dp_param = self.HDrawCluster(lc, rng_eng)

        self._dp_params[lc] = new_dp_param

    def update_aux_params(
            self, igal, curr, prop, lnp_gal, lnp_gal_proposal, rng_eng
    ):

        unif = uniform_distribution
        unif.random_state = rng_eng

        L0 = self.ScalingParamLnPosterior(curr, lnp_gal)
        L1 = self.ScalingParamLnPosterior(prop, lnp_gal_proposal)

        h = min(0., L1 - L0)
        u = unif.rvs(0, 1)

        if log(u) <= h:
            self._scale_params[igal] = prop
        else:
            self._scale_params[igal] = curr

    def update_hyperparameters(self, rng_eng):

        if self._sample_dp_precision:
            self.GibbsUpdateDPPrecision(rng_eng)

    def del_c_i(self, i):
        pass

    def DrawClassSelectorFromCondition(
            self, i, eta_i, lnp_gal, lnp_marg, rng_eng
    ):
        """

        Schneider sampling algorithm

        Parameters
        ----------
        i : _type_
            _description_
        eta_i : _type_
            _description_
        lnp_gal : _type_
            _description_
        lnp_marg : _type_
            _description_
        rng_eng : _type_
            _description_
        """

        # For code parsing, let's assume an example 'c' vector and enumerate
        # what happens to it in each line of code below.
        # Ex: for 6 galaxies, with 2 galaxies per class: c = [0, 0, 1, 1, 3, 3]
        label = self._latent_classes[i]

        # ----- Draw a new value for c_i
        # First copy all but the 'i'th latent class selectors
        # Ex: i = 2
        # Ex: c_minus_i = [0, 0, 1, 3, 3]
        # Ex: Here we have chosen a scenario where latent class '2' is empty
        status = self._latent_classes.RemoveLatentClass(i)

        if status > 0:
            self._dp_params.erase(label)

        # remove all duplicate elements from c_j
        # Ex: num_unique_cj = 3
        num_unique_cj = self._latent_classes.GetNumClusters()

        # We destroyed c_j, so create a new array with all latent class
        # selectors except for the 'i'th one.
        # These lines implement Eqns. (26, 27) of Schneider et al. (2015)
        # Ex: len(p)=4; p[0:3] = q_over_b(i=2, j=0:3) * n_minus_i
        # Ex: n_minus_i = [2, 1, 2]
        # Ex: p[4] = r_over_b(i=1) * dp_precision_
        p = [0]*(num_unique_cj + 1)
        for j in range(num_unique_cj):
            n_minus_i = self._latent_classes.GetNumGalInCluster(
                self._latent_classes.UniqueClusters(j)
            )
            p[j] = lnp_gal[j] * n_minus_i

        p[num_unique_cj] = lnp_marg * self._dp_precision

        # Normalize the 'p' array to sum to 1
        # This defines the normalization parameter 'b' in the functions
        # 'q_over_b' and 'r_over_b'.
        # (The name 'b' matches the notation of Neal 2000.)
        p_norm = 0.
        for j in range(num_unique_cj+1):
            p_norm += p[j]
        for j in range(num_unique_cj+1):
            p[j] /= p_norm

        # Draw an integer on the interval [0, num_unique_cj] with
        # probabilities p
        # Ex: Choose a value for c_draw = 2 (from interval [0, 4])
        c_draw = self.DrawLatentClassAssignement(p, num_unique_cj+1, rng_eng)

        # Re-assign c_draw to be the value of the latent class index at
        # position c_draw.
        # That is, c_draw is on [0, num_unique_cj], but the latent class
        # indices are defined on [0, ngal-1] so we need to index the
        # rank-ordered list of unique latent class indices to assign the
        # correct integer value for c_draw.
        # Note: This is different from the Python version of Thresher where we
        # relabeled the latent class indices at this stage. But this
        # implementation is cleaner.
        # Ex: c_minus_i unique clusters: [0, 1, 3]
        # Ex: c_draw = [0, 1, 3][2] = 3
        # Ex: The 'Note' just above means we leave unique clusters as [0, 1, 3]
        # Ex: rather than relabeling to [0, 1, 2]
        c_draw = self._latent_classes.UniqueClusters(c_draw)

        # Does c_draw match any existing values of c?
        # Ex: new_c = false because c_draw = 3, which already exists
        new_c = self._latent_classes.IsNewClass(c_draw)

        # ----- Re-assign all c values with the new labels
        # Ex: c_minus_i = [0, 0, 3, 1, 3, 3]
        self._latent_classes.InsertClassIndex(i, c_draw)

        # ----- Draw a new value for sigma^2 if the new c_i does not match any
        # other c's
        if new_c:
            new_dp_param = self.HDraw(i, eta_i, rng_eng)
            self._dp_params[c_draw] = new_dp_param

    def update_latent_param(self, i, p, lnp_gal, lnp_marg, rng_eng):
        self.DrawClassSelectorFromCondition(i, p, lnp_gal, lnp_marg, rng_eng)

    def InitOutputData(self, nsteps):
        self._alpha_out = {"alpha": [-99]*nsteps}
        self._scale_out = {"scale": [-99]*nsteps}
        if self._save_lc_labels:
            self._lc_out = {"latent_class": [-99]*nsteps}
        self._nlc_out = {"nlc": [-99]*nsteps}
        self._dpprec_out = {"dpprec": [-99]*nsteps}

    def write_to_hdf5(self, istep):
        pass

    def save_step(self, istep):

        n = min(self._n_lc_max, self.ngroups())

        # Intrinsic ellipticity variances - 1 for each latent class
        s_out = [0. for i in range(self._n_lc_max)]
        for i in range(n):
            s_out[i] = self._dp_params[i]
        self._alpha_out["alpha"][istep] = s_out

        # scale_params - 1 for each galaxy
        sc_out = [self._scale_params[igal] for igal in range(self._ngals)]
        self._scale_out["scale"][istep] = sc_out

        # latent class labels - 1 for each galaxy
        if self._save_lc_labels:
            lc_out = [
                self._latent_classes[igal] for igal in range(self._ngals)
            ]
            self._lc_out["latent_class"][istep] = lc_out

        # Counts per latent class
        n_in_class = [0 for i in range(self._n_lc_max)]
        for i in range(n):
            n_in_class[i] = self._latent_classes.GetNumGalInCluster(i)
        self._nlc_out["nlc"][istep] = n_in_class

        # DP precision hyperparameter
        if self._sample_dp_precision:
            self._dpprec_out["dpprec"][istep] = self._dp_precision

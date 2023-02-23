from galaxy_prior import EllipticityPriorGaussian
from galaxy_hyperprior_DP import GalaxyDistPriorDP
from shear_model import ShearModelConst

from configparser import ConfigParser
import h5py
import gc
from tqdm import tqdm

from math import exp, log
from scipy.stats import uniform as uniform_distribution
import numpy


N_LENS_PARAMS = 3
CURRENT = 0
PROPOSAL = 1


class Thresher():
    """
    Main Thresher class
    """

    def __init__(
            self,
            gal_prior: EllipticityPriorGaussian,
            gal_dist: GalaxyDistPriorDP,
            shear_model: ShearModelConst,
            seed: int,
            verbose: bool = False,
    ):
        self._gal_prior = gal_prior
        self._gal_dist = gal_dist
        self._shear_model = shear_model
        self._rng_eng = numpy.random.Generator(numpy.random.MT19937(seed))
        self._verbose = verbose

        self._NGALS = 0
        self._NSAMPLES = 0
        self._NPARAMS = 0
        self._NWALK = 0
        self._NSTEP = 0

        # Initialize Reaper input parsing indices to that for FakeReaper
        self._E1_NDX = 0
        self._E2_NDX = 1
        self._RA_NDX = 4
        self._DEC_NDX = 5
        self._LNR_NDX = -1
        self._LNF_NDX = -1
        self._LNPRIOR_NDX = 3

        self._sample_aux_params = False

        self._n_interim = 10
        self._lnp = -99999
        self._infile = ""

    def Allocate(self):
        """
        Must be called after Pick() method
        """

        if self._verbose:
            print("Thresher::Allocate")
            print(
                "NGALS:", self._NGALS,
                "NSAMPLES:", self._NSAMPLES,
                "NPARAMS:", self._NPARAMS
                )

        self._gal_dist.set_ngals(self._NGALS)
        self._gal_dist.Allocate()

        self._shear_model.reset_shear_acceptance_count()
        self._lnp = self.LnPosterior(CURRENT, False)
        if self._verbose:
            print("<Thresher> Initial lnp:", self._lnp)
            print("Finished Thresher::Allocate")

    def SetParameters(self, paramfile):

        if self._verbose:
            print("<Thresher> Setting hyperparameters from", paramfile)

        ini = ConfigParser()
        ini.read(paramfile)

        self._n_interim = ini.getint("sampling", "n_interim", fallback=10)
        self._THRESHER_OUTFILE_NAME = ini.get(
            "outputs",
            "THRESHER_OUTFILE_NAME",
            fallback="../dls/thresher_out.h5"
        )
        self._E1_NDX = ini.getint("inputs", "E1_NDX", fallback=0)
        self._E2_NDX = ini.getint("inputs", "E2_NDX", fallback=1)
        self._LNR_NDX = ini.getint("inputs", "LNR_NDX", fallback=-1)
        self._LNF_NDX = ini.getint("inputs", "LNF_NDX", fallback=-1)
        self._RA_NDX = ini.getint("inputs", "RA_NDX", fallback=4)
        self._DEC_NDX = ini.getint("inputs", "DEC_NDX", fallback=5)
        self._LNPRIOR_NDX = ini.getint("inputs", "LNPRIOR_NDX", fallback=3)

        self._sample_aux_params = ini.getboolean(
            "sampling", "sample_aux_params", fallback=False
        )

        self._gal_dist.SetParameters(ini, self._verbose)
        self._shear_model.SetParameters(ini)

        if self._verbose:
            print("--- <Thresher> Parameters set from INI file:")
            print("\tn_interim_:", self._n_interim)

    def Pick(self, infile, nburn, verbose=False):
        """
        Read the interim samples from file (in HDF5 format)
        param infile name of the input HDF5 file with interim samples of galaxy
        model parameters.

        Ref:
        http://www.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html

        Parameters
        ----------
        infile : _type_
            _description_
        nburn : _type_
            _description_
        verbose : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        self._infile = infile
        if self._verbose:
            print("<Thresher> reading ")

        dataset_name = "gals/samples"

        try:
            f = h5py.File(infile, 'r')
            dataset = f[dataset_name]

            rank = dataset.ndim
            shape = dataset.shape

            if self._verbose:
                print("rank", rank, "dims", shape)

            if rank == 4:
                self._NGALS = shape[0]
                self._NSTEP = shape[1]
                self._NWALK = shape[2]
                self._NPARAMS = shape[4]
                self._NSAMPLES = self._NSTEP * self._NWALK
            else:
                self._NGALS = shape[0]
                self._NSAMPLES = shape[1]
                self._NPARAMS = shape[2]

            samples_in = numpy.empty(
                (self._NGALS, self._NSAMPLES, self._NPARAMS),
                dtype=dataset.dtype,
            )
            dataset.read_direct(samples_in)

            # Copy only n_samples_ from the input to the final samples_ array
            assert self._n_interim < (self._NSAMPLES - nburn)
            ndx = uniform_distribution
            ndx.random_state = self._rng_eng
            self._samples = numpy.empty(
                (self._NGALS*self._n_interim*self._NPARAMS)
            )
            if self._n_interim <= self._NSAMPLES:
                for j in range(self._NGALS):
                    for i in range(self._n_interim):
                        # select a random index in the range [0, NSAMPLES_ - 1]
                        samp_index = int(
                            (self._NSAMPLES-1 - nburn)*ndx.rvs() + nburn
                        )
                        for k in range(self._NPARAMS):
                            self._samples[
                                k + self._NPARAMS*(i + self._n_interim*j)
                            ] = samples_in[j, samp_index, k]
            else:
                raise ValueError(
                    "Requested number of interim samples is too large"
                )

            self._NSAMPLES = self._n_interim
            del samples_in
            gc.collect()

            if verbose:
                for j in range(self._NGALS):
                    print("Galaxy", j+1, "\ne1, e2, lnpost, lnprior")
                    for i in range(self._NSAMPLES):
                        str = ""
                        for k in range(self._NPARAMS):
                            str += " {}".format(
                                self._samples[
                                    k+self._NPARAMS*(i + self._NSAMPLES*j)
                                ]
                            )
                        print(str)
                    print()

            # Pre-calculate the inverse interim prior weighting for drawing
            # samples later on
            self._inv_lnprior_norm = [0]*self._NGALS
            for igal in range(self._NGALS):
                self._inv_lnprior_norm[igal] = 0.
                for isamp in range(self._NSAMPLES):
                    self._inv_lnprior_norm[igal] += exp(
                        -self._samples[
                            self._LNPRIOR_NDX +
                            self._NPARAMS*(isamp + self._NSAMPLES*igal)
                        ]
                    )

            self.Allocate()
            f.close()
        except Exception as e:
            raise ValueError(
                f"This Exception occured while reading samples:\n{e}"
            )

        return 0

    def SetOutfile(self, outfile):
        self._THRESHER_OUTFILE_NAME = outfile

    def InitOutputData(self, nsteps):
        self._shear_model.InitOutputData(nsteps)
        self._gal_dist.InitOutputData(nsteps)

    def get_shear_acceptance_count(self):
        return self._shear_model.get_shear_acceptance_count()

    def get_num_groups(self):
        return self._gal_dist.ngroups()

    def get_e1(self, igal, isamp):
        return self._samples[
            self._E1_NDX + self._NPARAMS*(isamp + self._NSAMPLES*igal)
        ]

    def get_e2(self, igal, isamp):
        return self._samples[
            self._E2_NDX + self._NPARAMS*(isamp + self._NSAMPLES*igal)
        ]

    def get_lnr(self, igal, isamp):
        return 0.

    def get_lnF(self, igal, isamp):
        return 0.

    def get_ra(self, igal, isamp):
        return 0.

    def get_dec(self, igal, isamp):
        return 0.

    def get_lnp(self):
        return self._lnp

    def get_unlensed_params(self, p, igal, isamp):
        p[0] = self.get_e1(igal, isamp)
        p[1] = self.get_e2(igal, isamp)
        p[2] = self.get_lnr(igal, isamp)
        p[3] = self.get_lnF(igal, isamp)
        e1, e2, ln_r_e, ln_F = self._gal_prior.get_unlensed_params(
            p[0], p[1], p[2], p[3]
        )
        return e1, e2, ln_r_e, ln_F

    def LnLikeGalaxyGivenAlpha(self, igal, isamp_min=0, isamp_max=9999):

        if isamp_max == 9999:
            isamp_max = self._NSAMPLES

        interim_prior_ndx = self._LNPRIOR_NDX

        interim_prior = 0.
        res = 1e-20
        for isamp in range(isamp_min, isamp_max):
            if interim_prior_ndx > 0:
                interim_prior = self._samples[
                    interim_prior_ndx +
                    self._NPARAMS*(isamp+self._NSAMPLES*igal)
                ]

            e1 = self.get_e1(igal, isamp)
            e2 = self.get_e2(igal, isamp)
            lnp = self._gal_prior.eval(e1, e2)

            res += exp(lnp - interim_prior)

        lnjac = self._gal_dist.gat_lnjac_omega(igal)
        return log(res) - log(self._NSAMPLES) + lnjac

    def LnLikeAlpha(
            self, selector, weak_shear=False, isamp_min=0, isamp_max=9999
    ):
        """
        Evaluate the marginal likelihood via importance sampling of input
        samples

        Parameters
        ----------
        selector : _type_
            _description_
        weak_shear : bool, optional
            _description_, by default False
        isamp_min : int, optional
            _description_, by default 0
        isamp_max : int, optional
            _description_, by default 9999

        Returns
        -------
        _type_
            _description_
        """

        res = 0.
        for igal in range(self._NGALS):
            self._gal_prior.set_hyperprior_params(
                self._gal_dist.get_params(igal)
            )
            self._gal_prior.set_lensing_params(
                self._shear_model.get_params(igal, selector)
            )
            res += self.LnLikeGalaxyGivenAlpha(igal, isamp_min, isamp_max)

        return res

    def LnPosterior(self, selector, weak_shear=False):
        """
        Evaluate the log posterior for the shear

        Parameters
        ----------
        selector : _type_
            _description_
        weak_shear : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """

        lnpriors = 0.
        lnpriors += self._shear_model.LnPior(selector)
        lnlike = self.LnLikeAlpha(selector, weak_shear)
        lnpriors += self._gal_dist.LnPrior()

        return lnlike + lnpriors

    def update_latent_classes(self):

        # List of model params for a single galaxy
        p = [0]*self._gal_prior._N_GAL_PARAMS

        # Gibbs update(s) for galaxy distribution parameters
        for igal in range(self._NGALS):
            ngroups = self._gal_dist.ngroups()
            pr_gal = [0]*ngroups

            for ilc in range(ngroups):
                sigma_e_sq = self._gal_dist.get_params_for_group(igal, ilc)
                self._gal_prior.set_hyperprior_params(sigma_e_sq)
                self._gal_prior.set_lensing_params(
                    self._shear_model.GetReducedShear(igal, CURRENT)
                )
                pr_gal[ilc] = exp(self.LnLikeGalaxyGivenAlpha(igal))

            interim_prior = 1.
            pr_marg = 0.
            for ndx in range(self._NSAMPLES):
                if self._LNPRIOR_NDX > 0:
                    interim_prior = exp(self._samples[
                        self._LNPRIOR_NDX +
                        self._NPARAMS*(ndx+self._NSAMPLES*igal)
                    ])

                p[0], p[1], p[2], p[3] = self.get_unlensed_params(
                    p, igal, ndx
                )
                pr_marg += self._gal_dist.r_over_b(igal, p) / interim_prior
            pr_marg /= self._NSAMPLES

            self._gal_dist.update_latent_param(
                igal, p, pr_gal, pr_marg, self._rng_eng
            )
            del pr_gal
            gc.collect()

    def update_group_params(self):

        # List of model params for a single galaxy
        p = [0]*self._gal_prior._N_GAL_PARAMS

        unif = uniform_distribution
        unif.random_state = self._rng_eng

        # Loop over groups (i.e., latent classes) and update alpha for each
        for group_ndx in range(self._gal_dist.ngroups()):
            # The conditional posterior for alpha depends on sums of omega
            # params for each galaxy associated with the group.
            # Aggregate those values here.
            self._gal_dist.reset_interim_sample_aggregation()

            for ig in range(self._gal_dist.n_in_group(group_ndx)):
                igal = self._gal_dist.gal_index_from_group(ig, group_ndx)

                # Get an index for one interim sample
                u = unif.rvs(0, 1)
                prob = 0.
                isamp = 0
                for i in range(self._NSAMPLES):
                    prob += \
                        (1. / self._samples[
                            self._LNPRIOR_NDX +
                            self._NPARAMS*(i + self._NSAMPLES*igal)
                        ]) / self._inv_lnprior_norm[igal]
                    if u <= prob:
                        isamp = i
                        break
                p[0], p[1], p[2], p[3] = self.get_unlensed_params(
                    p, igal, isamp
                )
                self._gal_dist.aggregate_interim_samples(igal, group_ndx, p)
            self._gal_dist.update_group_params(group_ndx, self._rng_eng)

        # MH update scale parameters sequentially through every galaxy
        if self._sample_aux_params:
            for igal in range(self._NGALS):
                curr = self._gal_dist.get_aux_params(igal)
                self._gal_prior.set_hyperprior_params(
                    self._gal_dist.get_params(igal)
                )
                lnp_gal = self.LnLikeGalaxyGivenAlpha(igal)

                prop = self._gal_dist.propose_aux_param(igal, self._rng_eng)
                self._gal_prior.set_hyperprior_params(
                    self._gal_dist.get_params(igal)
                )
                lnp_gal_prop = self.LnLikeGalaxyGivenAlpha(igal)

                self._gal_dist.update_aux_params(
                    igal, curr, prop, lnp_gal, lnp_gal_prop, self._rng_eng
                )

    def Step(self, istep):

        # MH update for shear
        self._shear_model.propose()
        lnp1 = self.LnPosterior(PROPOSAL)
        self._lnp = self._shear_model.update(self._lnp, lnp1)

        # Gibbs updates for latent class selectors
        self.update_latent_classes()

        # Gibbs / MH updates for parameters per class
        self.update_group_params()

        # Hyperparameters for the galaxy distribution
        self._gal_dist.update_hyperparameters(self._rng_eng)

        # Recompute ln-posterior after galaxy dist. updates
        self._lnp = self.LnPosterior(CURRENT)

        # self._shear_model.write_to_hdf5()
        # self._gal_dist.write_to_hdf5()
        self._shear_model.save_step(istep, self._lnp)
        self._gal_dist.save_step(istep)


def do_sampling(
        thresh: Thresher,
        nsteps: int,
        verbose: bool,
):
    status = 0
    print("Sampling")

    try:
        thresh.InitOutputData(nsteps)

        for istep in tqdm(range(nsteps)):
            count = thresh.get_shear_acceptance_count() / (istep+1)
            if (istep % 10 == 0) & verbose:
                print("istep:", istep+1, "/", nsteps)
                print(
                    "\t lnp:", thresh.get_lnp(),
                    "[g accept. frac.:", count, "]",
                    "num clusters:", thresh.get_num_groups()
                )
            thresh.Step(istep)
    except Exception as e:
        print(f"This error occured while sampling:\n{e}")

    return status

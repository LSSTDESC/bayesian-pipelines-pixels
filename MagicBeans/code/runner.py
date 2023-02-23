from galaxy_prior import EllipticityPriorGaussian
from galaxy_hyperprior_DP import GalaxyDistPriorDP
from shear_model import ShearModelConst
from Thresher_C import Thresher, do_sampling
gal_prior =  EllipticityPriorGaussian(0, 0, 0, True)
gal_dist = GalaxyDistPriorDP(0.0001, 0.0001, 0.09)
shear_model = ShearModelConst()
thresh = Thresher(gal_prior, gal_dist, shear_model, 420, True)
thresh.SetParameters("../test/parameters/thresher_params_toymodel_python.ini")
status = thresh.Pick("../test/data/fake_reap_toymodel.h5", 100, False)
if status == 0:
    nsteps = 3000
    status = do_sampling(thresh, nsteps, True)
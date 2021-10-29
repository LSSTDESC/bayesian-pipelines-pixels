# bayesian-pipelines
Bayesian cosmological inference directly from pixels. 

## Pipeline stages 

There are multiple stages in an inference pipeline to constrain cosmological parameters from Rubin images. For some stages, there exist multiple methods in development for Bayesian inference. Other stages, do not yet have Bayesian models implemented and ready for exploration within DESC. Our goal is to implement and demonstrate Bayesian pipelines for a subset of the possible stages listed below, to begin to establish performance metrics for the Bayesian pipeline approach.

1. PSF measurement and interpolation
2. Galaxy image analysis
3. Shear inference
4. Photo-z inference
5. LSS inference
6. Cosmological parameter inference

## Generative model (v1)

1. Log-normal model of density and lensing convergence with a cosmology-dependent power spectrum
2. Redshifts asserted (no photo-z)
3. Lensing shear and convergence given as inputs to image simulations
4. Isolated (non-blended) galaxy images rendered in postage stamps using GalSim (and/or Lanusse differentiable implementation thereof)
5. Constant PSF, Sersic galaxy profiles, artificially small shape noise

This model will test Bayesian pipeline stages for: galaxy image analysis, shear inference, and cosmological parameter inference.
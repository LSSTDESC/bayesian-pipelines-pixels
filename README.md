# bayesian-pipelines-pixels
Bayesian inference of lensing shear directly from pixels. 

This project aims to demonstrate probabilistic analysis of simulated Rubin single-visit and coadd images to infer posterior weak lensing shear and convergence fields on the sky. The modeling will be tested on both forward simulated data and LSST DESC Data Challenge 2 (DC2) images.

We are building a Bayesian pipeline for weak lensing shear and convergence inference combining multiple existing DESC tools/projects. Our pipeline includes a generative model for galaxy images that is used for inference as well as training and validation of specific inference algorithms. The generative model starts from a fixed cosmology and redshift distribution, uses two-dimensional random (Gaussian or log-Normal) fields to build a correlated shear field, and uses `galsim` to draw and shear galaxies. The inference pipeline will consist of detection, deblending, shape inference, and shear inference stages. 

See the [Project Roadmap](https://github.com/LSSTDESC/bayesian-pipelines-pixels/issues/1) for details about the structure of the generative model and inference pipeline.

## Existing codes being interfaced here:

- [BLISS](https://github.com/prob-ml/bliss)
- [BFD](https://github.com/rearmstr/desc_bfd)
- [JIF](https://github.com/mdschneider/JIF)
- [MADNESS](https://portal.lsstdesc.org/DESCPub/app/PB/show_project?pid=251)


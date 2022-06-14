# Bayesian Pixel Pipeline

Bayesian inference of lensing shear directly from pixels.

This project aims to demonstrate probabilistic analysis of simulated Rubin single-visit and coadd images to infer posterior weak lensing shear and convergence fields on the sky. The modeling will be tested on both forward simulated data and LSST DESC Data Challenge 2 (DC2) images.

We are building a Bayesian pipeline for weak lensing shear and convergence inference combining multiple existing DESC tools/projects. Our pipeline includes a generative model for galaxy images that is used for inference as well as training and validation of specific inference algorithms. The generative model starts from a fixed cosmology and redshift distribution, uses two-dimensional random (Gaussian or log-Normal) fields to build a correlated shear field, and uses `galsim` to draw and shear galaxies. The inference pipeline will consist of detection, deblending, shape inference, and shear inference stages.

See the [Project Roadmap](https://github.com/LSSTDESC/bayesian-pipelines-pixels/issues/1) for details about the structure of the generative model and inference pipeline.

## Existing codes being interfaced here

- [BLISS](https://github.com/prob-ml/bliss)
- [BFD](https://github.com/rearmstr/desc_bfd)
- [JIF](https://github.com/mdschneider/JIF)
- [MADNESS](https://portal.lsstdesc.org/DESCPub/app/PB/show_project?pid=251)

## Install

1. Install [poetry](https://python-poetry.org/docs/), our package manager. Specifically for osx/linux:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. Follow the instructions that appear after running the above installation command. In particular make sure that you add `poetry` installation to your `PATH`. You might need to restart your terminal.

3. Git clone the repository
    ```bash
    git clone https://github.com/LSSTDESC/bayesian-pipelines-pixels
    ```

4. Install the dependencies with poetry.
    ```bash
    cd bayesian-pipelines-pixels
    poetry install
    ```

5. Launch virtual environment (anytime you work on the package)
    ```bash
    poetry shell
    ```

6. Package is now editable and importable within this virtual environment. Dependencies should be importable too.
    ```python
    import bpp
    import galsim
    ```

7. Ensure everything works correctly by running the tests
    ```bash
    pytest
    ```

**Important Note:** If the `poetry.lock` file gets updated in a pull request, that means that the dependecies of the project were updated. You will need to run `poetry install` to use the latest version of dependencies. When in doubt, run `poetry install` whenever the `main` branch has a new commit.

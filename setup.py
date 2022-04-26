from setuptools import setup

setup(
    name="bayes_pipelines_pixels",
    version="0.0.1",
    url="https://github.com/LSSTDESC/bayesian-pipelines-pixels",
    author="Ismael Mendoza, Michael Schneider, Bob Armstrong, Biswajit Biswas",
    author_email="imendoza@umich.edu",
    description="Bayesian inference of lensing shear directly from pixels.",
    packages=["bpp"],
    install_requires=["galsim", "numpy"],
)

# Demeter 0.2
![](demeter_logo_darkLight.png)
(DiffEoMorphic METamorphic Regristration)

Package status : alpha. see [Documentation](https://antonfrancois.github.io/Demeter_metamorphosis/)

Demeter_metamorphosis is a library intended to perform Metamorphosis on images.
It can perform efficient registration of images in 2D and 3D with or without topology differences.
We propose a flexible framework to perform Metamorphosis with differents modeling
choices. Are provided:
- The classic LDDMM registration method for registering image having the same topologies as described by Trouvé et al. or Beg's 2005 paper.
- The classic Metamorphosis registration method for mixing diffeomorphic transport with intensities additions as described by Trouvé et al.  2005 paper.
- A Weighted Metamorphosis registration method to control the intensity addition locally as described by me IN.
- A Constrained Metamorphosis registration method to control the intensity addition locally and guide the registration with a pre-computed vector field as described by me IN.
- and more to come...

The library is designed to be flexible and allow the user to define custom data cost functions, custom norms on the residual, custom vector fields, and custom geodesic integrators. Feel
free to contact me through github Issues if you have any questions or need help with the library.

Why the name Demeter ? Because in ancient Rome, Demeter was the goddess of agriculture.
Isn't it the perfect protection for or vectors fields ? 

## Requirement 

In this project depend on plural libraries, some quite basics like Numpy 


and some more complex. make sure that you have :
- [`torch 1.8`](https://pytorch.org/) or higher as we use the fft function 
- [`kornia`](https://pypi.org/project/kornia/).
- [`vedo 2021.0.5`](https://vedo.embl.es/) for 3D visualisation


Demeter plotting capabilities (i.e., functions start with plot_ and classes end with Display) require Matplotlib (>= 3.3.4). For running the examples Matplotlib >= 3.3.4 is required. A few examples require scikit-image >= 0.17.2, a few examples require pandas >= 1.2.0, some examples require seaborn >= 0.9.0 and plotly >= 5.14.0.

## Installation
At the moment, demeter-metamorphosis is available on linux and macOS only

### From pip
The last stable version of demeter_metamorphosis can be installed directly from pip with
```bash
pip install demeter
```
### From source
If you want the development version or consider contributing to the codebase,
you can also install scikit-shapes locally from a clone of the repository. 
First clone the repository with one of the github provided methods. For
example, with ssh:
```bash
git clone git@github.com:antonfrancois/Demeter_metamorphosis.git
```
I advise you to create a fresh virtual environment with conda or venv. With venv [more...](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments):
conda
```bash
conda create -n demeter_env python=3.12
conda activate demeter_env
```
OR
```bash
python3.12 -m venv "demeter_env"
source "demeter_env"/bin/activate
```
Then activate it and navigate to the cloned repository. You can install the package with
```bash
cd Your/path/to/Demeter_metamorphosis
pip install -e .
```
It can take up to 10 minutes depending on your internet connection, mainly because of torch and nvidia driver installation. 


## Jupyter Notebooks and examples.

You can see results and usage exemples in the `examples` folder or at the [Example Gallery](https://antonfrancois.github.io/Demeter_metamorphosis/auto_examples/index.html:


## Contact

You can email me at anton.francois [at] ens-paris-saclay.fr or check my website : [antonfrancois.github.io/](antonfrancois.github.io/)
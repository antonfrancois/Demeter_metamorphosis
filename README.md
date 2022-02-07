# Demeter 0.2
![](demeter_logo.png)
(DiffEoMorphic METamorphic Regristration)

**Warning** : This depot is still in progress.

You are seeing the evolution of a new python librairy, intended to solve 
diffeomoprhic registration problems using metamorphosis, LDDMM and 
derived methods.
For more details on the methods please read the joined PDF :
`Metamorphic_image_registration_afrancois_pgori_jglaunes`.

Why Demeter ? Because in ancient Rome, Demeter was the goddess of agriculture.
Isn't it the perfect protection for or vectors fields ? 

## Requirement 

In this project we use some libraries, some quite basics like :
- Numpy 
- math 
- matplotlib 
- time


Make sure that you have `torch 1.8` or higher for fft and `kornia` that you can 
download like indicated in the following links [(PyTorch)](https://pytorch.org/)
and [(kornia)](https://pypi.org/project/kornia/).

## Jupyter Notebooks

You can see results and usage exemples in the following jupyter notebooks

- **figs_geodesic_stability_lddmm.ipynb** : Figure from the article aiming to show the stability 
  of semi-Lagrangian schemes over Eulerian ones for LDDMM.
- **fig_lddmm_vs_metamorphoses.ipynb** : Figure from the article aiming to 
compare our implementation of LDDMM and Metamorphosis.
- **toyExample_weightedMetamorphosis.ipynb** : Examples and results of Weighted Metamorphosis method on a synthetic toy example image.


## Files

- **metamorphosis.py** : is containing the two classes for LDDMM/metamorphic
regristration 
    - **class** `Geodesic_integrator` : (abstract) is the class 'tool kit' for integrating over a geodesic
    - **class** `Optimise_geodesicShooting` : (abstract) is the class 'tool kit' for optimizing using geodesic shooting methods. It works in combinaison with a child class of `Geodesic_integrator`.
    - **class** `Metamophosis_path`: (derived from `Geodesic_integrator`) is made for integration over LDDMM or Metamorphosis geodesics.
    - **class** `Optimise_Metamorphosis`: (derived from `Optimise_geodesicShooting`) Optimize a `Metamorphosis_path` object.
    - **class** `Residual_norm_function`: (abstract) Generic 'tool kit' class for defining a custom norm on $z_0$
    - **class** `Residual_norm_identity`: (derived from `Residual_norm_function`) definition of $\|\sqrt{M_t}z_t\|_{L_2}^2 = \lbrace z_t M_t ,z_t \rbrace$
    - **class** `Residual_norm_borderBoost`: (derived from `Residual_norm_function`) Experimental
    - **class** `Weighted_meta_path`: (derived from `Geodesic_integrator`) is made for integration with Weighted Metamorphosis method. It is defined with a function derived from an child object of `Residual_norm_function`.
    - **class** `Weighted_optim`: (derived from `Optimise_geodesicShooting`) Optimize a `Weighted_meta_path` object.
      
- **my_optim.py** : is containing the gradient descent class.

- **my_toolbox.py** : is containing generic utilitary fonctions 

- **my_torchbox.py** : is containing utilitary function compatible with PyTorch 
such as: 
  - wrapper for `kornia` and `PyTorch` utilitary functions.
  - function for plotting.
    
- **fft_conv.py** : function from [Github : fkodom](https://github.com/fkodom/fft-conv-pytorch)

- **reproducing_kernels.py** : Kornia inerited functions to use filter convolution
with fft_conv. 

- **constants.py** : Files with different constants reused at different locations


## Contact

You can find my contact on my website : http://helios.mi.parisdescartes.fr/~afrancoi/
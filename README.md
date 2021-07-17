# Demeter 0.1
![](demeter_logo.png)
(DiffEoMorphic METamorphic Regristration)

**Warning** : This depot is still in progress.

You are seeing the birth of a new python librairy, intended to solve 
diffeomoprhic registration problems using metamorphosis and LDDMM.
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

## Files

- **metamorphosis.py** : is containing the two classes for LDDMM/metamorphic
regristration 
    - **class** `metamorphic_path`: is the class performing geodesic shootin
    - **class** `grad_descent_metamorphosis` : is the class instantiating 
      `metamorphic_path` and applying gradient descent with the cost stated in 
      the article.
      
- **my_optim.py** : is containing the gradient descent class.

- **my_toolbox.py** : is containing generic utilitary fonctions 

- **my_torchbox.py** : is containing utilitary function compatible with PyTorch 
such as: 
  - wrapper for `kornia` and `PyTorch` utilitary functions.
  - function for plotting.
    
- **fft_conv.py** : function from [Github : fkodom](https://github.com/fkodom/fft-conv-pytorch)

- **reproducing_kernels.py** : Kornia inerited functions to use filter convolution
with fft_conv. 
  
## Contact

You can find my contact on my website : http://helios.mi.parisdescartes.fr/~afrancoi/
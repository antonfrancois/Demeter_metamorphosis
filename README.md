# Demeter 0.1

(DiffEoMorphic METamorphic Regristration)

**Warning** : This depot is still in progress, Jupyter notebooks will be provided
soon.

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

Make sure that you have `torch 1.7.1` or higher and `kornia` that you can 
download like indicated [Here] and [here].


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
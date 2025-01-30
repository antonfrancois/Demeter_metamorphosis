Installation
============

So far, demeter-metamorphosis is only available for linux and macOS, if you are a
windows user, you can consider using
`wsl <https://learn.microsoft.com/en-us/windows/wsl/about>`_.

With pip
--------

TBA

From source
-----------

First, install `demeter_metamorphosis` ,you need to clone the `GitHub repository <https://github.com/antonfrancois/Demeter_metamorphosis>`_:.

You can do this by running the following command on a terminal,
navigate to the directory and run

.. code-block:: bash

    git clone git@github.com:antonfrancois/Demeter_metamorphosis.git

I advise you to create a fresh virtual environment with conda or venv.  :

With venv

.. code-block:: bash

    conda create -n demeter_env python=3.12
    conda activate demeter_env

OR with conda

.. code-block:: bash

    python3.12 -m venv "demeter_env"
    source "demeter_env"/bin/activate

Basic usage
*************
To start using demeter not wanting to get do deep in the functionalities, you can install the package locally with pip.

.. code-block:: bash

    cd Your/path/to/Demeter_metamorphosis
    pip install .

For developers
**************

If you want the development version or consider contributing to the codebase,
you can install it with access to the source code.

.. code-block:: bash

    cd Your/path/to/Demeter_metamorphosis
    pip install -e .

It can take up to 10 minutes depending on your internet connection, mainly because of torch and nvidia driver installation.
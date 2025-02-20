Installation
============

So far, demeter-metamorphosis is only available for linux and macOS, if you are a
windows user, you can consider using
`wsl <https://learn.microsoft.com/en-us/windows/wsl/about>`_.

WSL setup (for Windows users):
**********
WSL is a compatibility layer for running Linux binary executables natively on Windows 10 and Windows Server 2019. It is a feature of the Windows operating system that allows you to run a Linux distribution directly on Windows without using a virtual machine or container.
So first install WSL and then you can follow the Unix installation steps.

Install wsl:

.. code-block:: bash

    wsl --install

You might need to restart your computer to finish the installation process.

Open a terminal wsl. After the installation is completed, open wsl, set up a password.

.. code-block:: bash

    sudo apt update
    sudo apt install python3-pip
    sudo apt install python3.12-venv

For Unix systems, (Linux and macOS)
*****************************************

With pip
--------

TBA

From source
-----------

First, install `demeter_metamorphosis`, you need to clone the `GitHub repository <https://github.com/antonfrancois/Demeter_metamorphosis>`_:.

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
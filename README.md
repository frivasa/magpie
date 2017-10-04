# This is MAGPIE

A small handy tool for editing runs for MESA ([http://mesa.sourceforge.net/](http://mesa.sourceforge.net/ "MESA's Homepage")) from the comfort of a jupyter notebook.

# What can it do?
1. Display inlists as tables with *qgrid* for easy editing
1. Read defaults to see and browse parameters from MESA
1. Write inlists to your workfolder
1. Compile your workfolder

# Ok, how do I do all this?
First, make sure you have all the required packages. (all of these are pip-able)

## Required packages (as of 17/10/01)
* numpy
* pandas
* jupyter
* qgrid (specific version so that tables are editable)
   * `pip install qgrid==1.0.0b3`   
   * `jupyter nbextension enable --py --sys-prefix qgrid`   

Then, import magpie.py to your notebook and that's it.

## Instalation
Yes, you can run
* `git clone https://github.com/frivasa/magpie.git`

but since the tool is lightweight, you can just download or copy the magpie.py file directly to your working directory, and the notebook example to get started quickly.


## Crash course
Use the included notebook or see [this .io article](https://frivasa.github.io/magpie-examples.html) to get a feel of how to use it.

# semantic-design-languages
Code for supporting our NSF SYS Proposal and related publications

![Alt text](/glass.png)
![Alt text](/airfoil.png)

Edit experiment configurations in config.ini

To perform the embedding and synthesize new shapes: python training.py

To synthesize new shapes using trained models: python synthesis.py

The settings of the kernel PCA and autoencoders are in the configuration files in './hp-opt/hp\_\<example name\>\_\<noise scale\>.ini'

We use [pySMAC](http://pysmac.readthedocs.io/en/latest/#) for hyperparameter optimization of kernel PCA and autoencoders

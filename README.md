# Design Manifolds Capture the Intrinsic Complexity and Dimension of Design Spaces
Experiment code associated with our JMD paper: "[Design Manifolds Capture the Intrinsic Complexity and Dimension of Design Spaces](http://mechanicaldesign.asmedigitalcollection.asme.org/article.aspx?articleid=2610207)"

![Alt text](/design_manifolds.png)

Edit experiment configurations in config.ini

To perform the embedding and synthesize new shapes:
```
python training.py
```

To synthesize new shapes using trained models:
```
python synthesis.py
```

The settings of the kernel PCA and autoencoders are in the configuration files:
```
./hp-opt/hp_<example name>_<noise scale>.ini
```

We use [pySMAC](http://pysmac.readthedocs.io/en/latest/#) for hyperparameter optimization of kernel PCA and autoencoders.

The code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Chen W, Fuge M, Chazan J. Design Manifolds Capture the Intrinsic Complexity and Dimension of Design Spaces. ASME. J. Mech. Des. 2017;139(5):051102-051102-10. doi:10.1115/1.4036134.

    @article{chen2017design,
      title={Design Manifolds Capture the Intrinsic Complexity and Dimension of Design Spaces},
      author={Chen, Wei and Fuge, Mark and Chazan, Jonah},
      journal={Journal of Mechanical Design},
      volume={139},
      number={5},
      pages={051102-051102-10},
      year={2017},
      publisher={American Society of Mechanical Engineers}
    }

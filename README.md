Caffe for FlowNet2 
==================

This is the release of:
 - the CVPR 2017 version of FlowNet2.0

It comes as a fork of the caffe master branch and with trained networks,
as well as examples to use and train them.

License and Citation
====================

All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. When using the code in your research work, please cite the following paper:

    @InProceedings{IMKDB17,
      author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
      title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = "Jul",
      year         = "2017",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
    }

Compiling
=========

First compile caffe, by configuring a

    "Makefile.config" (example given in Makefile.config.example)

then make with 

    $ make -j 5 all tools pycaffe 


Running 
=======

(this assumes you compiled the code sucessfully) 

IMPORTANT: make sure there is no other caffe version in your python and 
system paths and set up your environment with: 

    $ source set-env.sh 

This will configure all paths for you. Then go to the model folder 
and download models: 

    $ cd models 
    $ ./download-models.sh 
 
Running a FlowNet on a single image pair ($net is a folder in models): 

    $ run-flownet.py /path/to/$net/$net_weights.caffemodel[.h5] \
                     /path/to/$net/$net_deploy.prototxt.template \ 
                     x.png y.png z.flo 

(where x.png and y.png are images and z.flo is the output file) 

Running  a FlowNet on lots of image pairs: 

    $ run-flownet-many.py /path/to/$net/$net_weights.caffemodel[.h5] \ 
                          /path/to/$net/$net_deploy.prototxt.template \
                           list.txt 

(where list.txt contains lines of the form "x.png y.png z.flo") 

NOTE: If you want to compute many flows, this option is much faster since 
caffe and the net are loaded only once. 


Training
========

(this assumes you compiled the code sucessfully) 

First you need to download and prepare the training data. For that go to the data folder: 

    $ cd data 

Then run: 

    $ ./download.sh 
    $ ./make-lmdbs.sh 

(this will take some time and quite some disk space) 

Then set up your network for training ($net is a folder in models):
	
    $ cd /path/to/$net 
    $ cp ../solver_S_<type>.prototxt solver.prototxt 
    $ cp $net_train.prototxt.template train.prototxt 
    # Edit train.prototxt and make sure all settings are correct 
    $ caffe train --solver solver.prototxt 

IMPORTANT: Edit train.prototxt to use your selected dataset and 
make sure the correct parts of the network are enabled by setting/adding
loss weights and blob learning rates. 

NOTE: The training templates include augmentation, during which an affine 
transformation is applied to a crop from the input immages. For training we 
use different batch sizes for each resolution:  

FlyingChairs: 		448 x 320 (batch size 8)
ChairsSDHom:		448 x 320 (batch size 8)
FlyingThings3D:		768 x 384 (batch size 4) 




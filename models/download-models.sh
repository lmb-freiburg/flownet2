#!/bin/bash 

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/resources/binaries/flownet2/flownet2-models.tar.gz
tar xvzf flownet2-models.tar.gz       

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/resources/binaries/flownet2/flownet2-models-kitti.tar.gz
tar xvzf flownet2-models-kitti.tar.gz       

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/resources/binaries/flownet2/flownet2-models-sintel.tar.gz
tar xvzf flownet2-models-sintel.tar.gz       

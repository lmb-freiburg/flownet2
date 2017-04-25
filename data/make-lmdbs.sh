#!/bin/bash 

../build/tools/convert_imageset_and_flow.bin FlyingChairs_release.list FlyingChairs_release_lmdb 0 lmdb 
../build/tools/convert_imageset_and_disparity.bin FlyingThings3D_release_TEST.list FlyingThings3D_release_TEST_lmdb 0 lmdb
../build/tools/convert_imageset_and_disparity.bin FlyingThings3D_release_TRAIN.list FlyingThings3D_release_TRAIN_lmdb 0 lmdb

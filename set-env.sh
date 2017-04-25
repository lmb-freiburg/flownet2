#!/bin/bash 

CAFFE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

RELEASE_PATH="$CAFFE_PATH/build" 

export PYTHONPATH="$CAFFE_PATH/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$RELEASE_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$RELEASE_PATH/tools:$RELEASE_PATH/scripts:$PATH"
export CAFFE_BIN="$RELEASE_PATH/tools/caffe"


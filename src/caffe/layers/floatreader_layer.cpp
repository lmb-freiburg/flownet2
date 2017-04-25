// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/float_reader_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {
  
template <typename Dtype>
void FloatReaderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void FloatReaderLayer<Dtype>::ReadData()
{
    if(data_) return;

    readFloatFile(this->layer_param_.reader_param().file(),data_,dataXSize_,dataYSize_,dataZSize_);
}

template <typename Dtype>
void FloatReaderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(top.size(), 1) << "FLOATREADER needs one output";

    ReadData();

    top[0]->Reshape(this->layer_param_.reader_param().num(),dataZSize_,dataYSize_,dataXSize_);
}

template <typename Dtype>
void FloatReaderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    assert(data_);
    for(int n=0; n<this->layer_param_.reader_param().num(); n++)
        memcpy(top[0]->mutable_cpu_data()+n*dataXSize_*dataYSize_*dataZSize_,data_,sizeof(float)*dataXSize_*dataYSize_*dataZSize_);
}

INSTANTIATE_CLASS(FloatReaderLayer);
REGISTER_LAYER_CLASS(FloatReader);

}  // namespace caffe

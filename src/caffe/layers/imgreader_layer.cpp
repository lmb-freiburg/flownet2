// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/img_reader_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output.hpp"
#include "caffe/data_transformer.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {
  
template <typename Dtype>
void ImgReaderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void ImgReaderLayer<Dtype>::ReadData()
{
    if(data_.count()) return;

    cv::Mat cv_img = ReadImageToCVMat(this->layer_param_.reader_param().file());
    CHECK(cv_img.data) << "Could not load " << this->layer_param_.reader_param().file();

    DataTransformer<Dtype> data_transformer(this->layer_param_.transform_param(), this->phase_);
    vector<int> top_shape = data_transformer.InferBlobShape(cv_img);

    data_.Reshape(top_shape);
    data_transformer.Transform(cv_img, &data_);
}

template <typename Dtype>
void ImgReaderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(top.size(), 1) << "ImgReader needs one output";

    ReadData();

    top[0]->Reshape(this->layer_param_.reader_param().num(),data_.channels(),data_.height(), data_.width());
}

template <typename Dtype>
void ImgReaderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    assert(data_.count());

    int num = this->layer_param_.reader_param().num();
    int size = data_.count();
    assert(top[0]->count()/size == num);

    for(int n=0; n<num; n++)
        memcpy(top[0]->mutable_cpu_data()+n*size,data_.cpu_data(),sizeof(float)*size);
}

INSTANTIATE_CLASS(ImgReaderLayer);
REGISTER_LAYER_CLASS(ImgReader);

}  // namespace caffe

// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/flow_augmentation_layer.hpp"

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
void FlowAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_GT(this->layer_param_.augmentation_param().crop_width(),0) << "Please enter crop width if you want to perform augmentation";
    CHECK_GT(this->layer_param_.augmentation_param().crop_height(),0) << "Please enter crop height if you want to perform augmentation";
    this->layer_param_.set_reshape_every_iter(false);
    LOG(WARNING) << "FlowAugmentationLayer only runs Reshape only on setup";
}

template <typename Dtype>
void FlowAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  CHECK_EQ(bottom.size(), 3) << "Flow augmentation layer takes three input blobs: FlowField, Img1TransfParams, Img2TransfParams";
  CHECK_EQ(top.size(), 1) << "Flow augmentation layer outputs one output blob: Augmented Flow";

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  //const int height = bottom[0]->height();
  //const int width = bottom[0]->width();

  CHECK_EQ(channels, 2) << "Flow data must have two channels";

  cropped_width_ = this->layer_param_.augmentation_param().crop_width();
  cropped_height_ = this->layer_param_.augmentation_param().crop_height();

  (top)[0]->Reshape(num,channels, cropped_height_, cropped_width_);
  
  
  //test_coeffs_.ReshapeLike(*bottom[1]);
  //test_coeffs_.ShareData(*bottom[1]); //reuse
  
  // Set up coeff blobs
  all_coeffs1_.ReshapeLike(*bottom[1]);
  all_coeffs2_.ReshapeLike(*bottom[2]);

  // How many params exist in general?
  AugmentationCoeff coeff;
  num_params_ = coeff.GetDescriptor()->field_count();

  // = Coeff transformation matrix cache for one batch
  coeff_matrices1_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tTransMat)));
  coeff_matrices2_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tTransMat)));

  
}

template <typename Dtype>
void FlowAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{  
    LOG(FATAL) << "Forward CPU Augmentation not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(FlowAugmentationLayer);
#endif

INSTANTIATE_CLASS(FlowAugmentationLayer);
REGISTER_LAYER_CLASS(FlowAugmentation);


}  // namespace caffe

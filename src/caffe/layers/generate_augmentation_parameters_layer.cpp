// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/generate_augmentation_parameters_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {
  
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}  

template <typename Dtype>
void GenerateAugmentationParametersLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->layer_param_.set_reshape_every_iter(false);
  LOG(WARNING) << "GenerateAugmentationParametersLayer only runs Reshape only on setup";
}

template <typename Dtype>
void GenerateAugmentationParametersLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  aug_ = this->layer_param_.augmentation_param();
  discount_coeff_schedule_ = this->layer_param_.coeff_schedule_param();
  
  // There is the following convention: 
  //   if there is just one blob given, the layer just generates the parameters with num equal to num of this blob
  //   if there are three blobs, it adds the generated parameters to the given ones and checks if the transformed image fits in the original one
  CHECK(bottom.size() == 1 || bottom.size() == 3) << "Generate augmentation parameters layer takes one (any blob from which it can take num and potentially original image size) or three (aug params, orig data, augmented data) input blobs.";
  CHECK_EQ(top.size(), 1) << "Generate augmentation parameters layer outputs one output blob.";
  
  // = Coeff stuff
  AugmentationCoeff coeff;
  num_params_ = coeff.GetDescriptor()->field_count();
  
  mode_ = aug_.mode();
  if (bottom.size() == 0)
    mode_ = "regenerate";
  if (bottom.size() == 1)
    if (bottom[0]->width() > 1 || bottom[0]->height() > 1)
      mode_ = "regenerate";
    
  // There are three modes: 
//     "add" - the generated parameters are added to the given ones; 
//     "replace" - the generated parameters replace the given ones (but the parameters which are not being generated are left as is); 
//     "regenerate" - whole parameter vector is set to zero and then the new parameters are generated
    
  LOG(INFO) << "mode: " << mode_;  
 
  if (bottom.size() >= 1)
    num_ = bottom[0]->num();
  else
    num_ = aug_.num();
  
  if (bottom.size() == 3) {
    cropped_width_ = (bottom)[2]->width();
    cropped_height_ = (bottom)[2]->height();     
    bottomwidth_ = (bottom)[1]->width();
    bottomheight_ = (bottom)[1]->height();
  } else {
    CHECK(aug_.has_crop_width() && aug_.has_crop_height()) << "Need crop_width and crop_height if there is no blob specifying these";
    cropped_width_ = aug_.crop_width();
    cropped_height_ = aug_.crop_height();
    if (bottom.size() == 1) {
      if (bottom[0]->width() > 1 || bottom[0]->height() > 1) {
        bottomwidth_ = bottom[0]->width();
        bottomheight_ = bottom[0]->height();
      } else {
        CHECK(aug_.has_bottomwidth() && aug_.has_bottomheight()) << "Need bottomwidth and bottomheight if there is no blob specifying these";
        bottomwidth_ = aug_.bottomwidth();
        bottomheight_ = aug_.bottomheight();
      }
    }
  }
  
  
  CHECK_GE(num_, 1) << "Must provide num with a bottom blob or in the prototxt";
  
//   LOG(INFO) << "mode=" << mode_ << ", gen_spatial_transform=" << gen_spatial_transform_ << ", gen_chromatic_transform=" << gen_chromatic_transform_;
  
  // Prepare output blob
  (top)[0]->Reshape(num_, num_params_, 1, 1); 
    
  num_iter_ = 0;  
}

template <typename Dtype>
void GenerateAugmentationParametersLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
  
  LOG(FATAL) << "CPU forward pass not implemented";

}

#ifdef CPU_ONLY
STUB_GPU(GenerateAugmentationParametersLayer);
#endif

INSTANTIATE_CLASS(GenerateAugmentationParametersLayer);
REGISTER_LAYER_CLASS(GenerateAugmentationParameters);

}  // namespace caffe

// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/data_augmentation_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

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
void DataAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  // TODO This won't work when applying a net to images of size different from what the net was trained on
  aug_ = this->layer_param_.augmentation_param();
  this->layer_param_.set_reshape_every_iter(false);
  LOG(WARNING) << "DataAugmentationLayer only runs Reshape on setup";
  if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping data mean blob initialization";
  else {
    if (aug_.recompute_mean()) {
      LOG(INFO) << "Recompute mean";
      this->blobs_.resize(3);
      this->blobs_[1].reset(new Blob<Dtype>());
      this->layer_param_.add_param();
      this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_lr_mult(0.);
      this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_decay_mult(0.);
      this->blobs_[2].reset(new Blob<Dtype>());
      this->layer_param_.add_param();
      this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_lr_mult(0.);
      this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_decay_mult(0.);      
    } 
    else {  
      LOG(INFO) << "Do not recompute mean";
      this->blobs_.resize(1);
    }
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));      
    // Never backpropagate
    this->param_propagate_down_.resize(this->blobs_.size(), false);
    this->layer_param_.add_param();
    this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_lr_mult(0.);
    this->layer_param_.mutable_param(this->layer_param_.param_size()-1)->set_decay_mult(0.); 
//     LOG(INFO) << "DEBUG: this->layer_param_.param_size()=" << this->layer_param_.param_size();
//     LOG(INFO) << "DEBUG: Writing layer_param";
    WriteProtoToTextFile(this->layer_param_, "/misc/lmbraid17/sceneflownet/dosovits/matlab/test/message.prototxt");
//     LOG(INFO) << "DEBUG: Finished writing layer_param";
  } 
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    LOG(WARNING) << "Reshape of Augmentation layer should only be called once? Check this";
    CHECK_GE(bottom.size(), 1) << "Data augmentation layer takes one or two input blobs.";
    CHECK_LE(bottom.size(), 2) << "Data augmentation layer takes one or two input blobs.";
    CHECK_GE(top.size(), 1) << "Data augmentation layer outputs one or two output blobs.";
    CHECK_LE(top.size(), 2) << "Data augmentation layer outputs one or two output blobs.";

    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();

    output_params_=(top.size()>1);
    input_params_=(bottom.size()>1);
    aug_ = this->layer_param_.augmentation_param();
    discount_coeff_schedule_ = this->layer_param_.coeff_schedule_param();

    // Dimensions
    do_cropping_ = (aug_.has_crop_width() && aug_.has_crop_height());
    if (!do_cropping_)
    {
        cropped_width_ = width;
        cropped_height_ = height;
        LOG(WARNING) << "Please enter crop size if you want to perform augmentation";
    }
    else
    {
        cropped_width_ = aug_.crop_width();    CHECK_GE(width, cropped_width_)   << "crop width greater than original";
        cropped_height_ = aug_.crop_height();  CHECK_GE(height, cropped_height_) << "crop height greater than original";
    }

    // Allocate output
    top[0]->Reshape(num, channels, cropped_height_, cropped_width_);

    // Coeff stuff
    AugmentationCoeff coeff;
    num_params_ = coeff.GetDescriptor()->field_count();

    // If this layer is given coefficients from another augmentation layer, take this blob (same ref)
    if (input_params_) {
        LOG(INFO) << "Receiving " << num_params_ << " augmentation params";
        all_coeffs_.ReshapeLike(*bottom[1]);
    } else
        all_coeffs_.Reshape(num, num_params_, 1, 1); //create

    if (output_params_) {
        top[1]->ReshapeLike(all_coeffs_);
        LOG(INFO) << "Emitting " << num_params_ << " augmentation params";
    }

    // Coeff transformation matrix cache for one batch
    coeff_matrices_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tTransMat)));
    
    coeff_chromatic_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tChromaticCoeffs)));
    coeff_chromatic_eigen_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tChromaticEigenCoeffs)));
    coeff_effect_.reset(new SyncedMemory(num * sizeof(typename AugmentationLayerBase<Dtype>::tEffectCoeffs)));

    chromatic_eigenspace_.reset(new SyncedMemory(sizeof(typename AugmentationLayerBase<Dtype>::tChromaticEigenSpace)));

    // Data mean computation
    if (aug_.recompute_mean()) {
      ones_.Reshape(1, 1, cropped_height_, cropped_width_);
      caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());
      this->blobs_[1]->Reshape(1, channels, cropped_height_, cropped_width_);
      this->blobs_[2]->Reshape(1, channels, 1, 1);
    }
    else if(aug_.mean().size()==3 && !aug_.mean_per_pixel())
    {
      ones_.Reshape(1, 1, cropped_height_, cropped_width_);
      caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());

      LOG(INFO) << "Using predefined per-pixel mean from proto";
      pixel_rgb_mean_from_proto_.Reshape(1,3,1,1);
      for(int i=0; i<3; i++)
          pixel_rgb_mean_from_proto_.mutable_cpu_data()[i]=aug_.mean().Get(i);
    }
    
    noise_.reset(new SyncedMemory(top[0]->count() / top[0]->num() * sizeof(Dtype)));

    *(this->blobs_[0]->mutable_cpu_data()) = 0;
    
//     LOG(INFO) << "DEBUG: Reshape done";
}


template <typename Dtype>
void DataAugmentationLayer<Dtype>::adjust_blobs(vector<Blob<Dtype>* > blobs)
{
    if (aug_.recompute_mean() > 0 && blobs.size() >= 2) {
      LOG(INFO) << "Data augmentation layer: adjusting mean blobs";
      CHECK_EQ(this->blobs_[1]->channels(), blobs[1]->shape(1));
      bool same_size = (this->blobs_[1]->width() == blobs[1]->shape(3)) &&
                       (this->blobs_[1]->height() == blobs[1]->shape(2));
      int channels = this->blobs_[1]->channels();
      int area = this->blobs_[1]->height() * this->blobs_[1]->width();
      int source_area = blobs[1]->shape(2) * blobs[1]->shape(3);
      this->blobs_[0]->CopyFrom(*blobs[0],false,true);
      LOG(INFO) << "recovered iteration count " << this->blobs_[0]->cpu_data()[0];
      if (this->layer_param().augmentation_param().mean_per_pixel()==false)
      {
             // If using RGB mean, copy only RGB values (blob 2)
             Blob<Dtype> tmp; tmp.CopyFrom(*blobs[2], false,true);
             caffe_gpu_memcpy(this->blobs_[2]->count()*sizeof(float), tmp.mutable_gpu_data(), this->blobs_[2]->mutable_gpu_data());

             Dtype* data_mean_per_channel_cpu = this->blobs_[2]->mutable_cpu_data();
             for(int i=0; i<this->blobs_[2]->count(); i++)
                 LOG(INFO) << "recovered mean value " << data_mean_per_channel_cpu[i];
      }
      else
      {
          if (same_size) {
            this->blobs_[1]->CopyFrom(*blobs[1],false,true);
            caffe_cpu_gemv(CblasNoTrans, channels, area, Dtype(1)/Dtype(area),
                           this->blobs_[1]->cpu_data(), ones_.cpu_data(), Dtype(0), this->blobs_[2]->mutable_cpu_data());
          } else {
            Blob<Dtype> tmp_mean;
            Blob<Dtype> tmp_ones;
            tmp_mean.CopyFrom(*blobs[1],false,true);
            tmp_ones.Reshape(1, 1, tmp_mean.height(), tmp_mean.width());
            caffe_set(tmp_ones.count(), Dtype(1), tmp_ones.mutable_cpu_data());
            caffe_cpu_gemv(CblasNoTrans, channels, source_area, Dtype(1)/Dtype(source_area),
                           tmp_mean.cpu_data(), tmp_ones.cpu_data(), Dtype(0), this->blobs_[2]->mutable_cpu_data());
            caffe_cpu_gemm(CblasNoTrans, CblasTrans, channels, area, 1,
                           Dtype(1), this->blobs_[2]->mutable_cpu_data(), this->ones_.cpu_data(), Dtype(0), this->blobs_[1]->mutable_cpu_data());
          }
      }
    }
    if (blobs.size() < 2)
      LOG(INFO) << "Data augmentation layer: no blobs to copy";
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
   	LOG(FATAL) << "Forward CPU Augmentation not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(DataAugmentationLayer);
#endif

INSTANTIATE_CLASS(DataAugmentationLayer);
REGISTER_LAYER_CLASS(DataAugmentation);

}  // namespace caffe

// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/resample_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define min(a,b) ((a<b)?a:b)

namespace caffe {

template <typename Dtype>
void ResampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    if(this->layer_param().resample_param().type() != ResampleParameter_ResampleType_CUBIC
            && this->layer_param().resample_param().type() != ResampleParameter_ResampleType_LINEAR
            && this->layer_param().resample_param().type() != ResampleParameter_ResampleType_NEAREST)
        LOG(FATAL) << "ResampleLayer: only CUBIC, LINEAR and NEAREST interpolation is supported for now";
}

template <typename Dtype>
void ResampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  this->layer_param_.set_reshape_every_iter(false);
  LOG(WARNING) << "ResampleLayer only runs Reshape on setup";
  
  CHECK_GE(bottom.size(), 1);
  CHECK_LE(bottom.size(), 2);
  CHECK_EQ(top.size(), 1);
  
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();

  int top_width;
  int top_height;

  if (bottom.size() == 1) {
    top_height = this->layer_param_.resample_param().height();
    top_width = this->layer_param_.resample_param().width();
  } else {
    top_height = bottom[1]->height();
    top_width = bottom[1]->width();
  }

  CHECK_GE(top_height, 1) << "ResampleLayer must have top_height > 0";
  CHECK_GE(top_width, 1) << "ResampleLayer must have top_width > 0";

  int count = top_width * top_height * channels * num;
  
  top[0]->Reshape(num, channels, top_height, top_width);
  CHECK_EQ(count, top[0]->count());
}

template <typename Dtype>
void ResampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  LOG(FATAL) << "ResampleLayer: CPU Forward not yet implemented.";
}

template <typename Dtype>
void ResampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  LOG(FATAL) << "ResampleLayer cannot do backward.";
}

INSTANTIATE_CLASS(ResampleLayer);
REGISTER_LAYER_CLASS(Resample);

}  // namespace caffe

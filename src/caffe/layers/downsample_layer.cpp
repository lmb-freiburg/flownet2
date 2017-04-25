// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/downsample_layer.hpp"
#include "caffe/util/math_functions.hpp"


#define min(a,b) ((a<b)?a:b)

namespace caffe {

template <typename Dtype>
void DownsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void DownsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  this->layer_param_.set_reshape_every_iter(false);
  LOG(WARNING) << "DownsampleLayer only runs Reshape on setup";
  
  CHECK_GE(bottom.size(), 1);
  CHECK_LE(bottom.size(), 2);
  CHECK_EQ(top.size(), 1);
  
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();

  if (bottom.size() == 1) {
    top_height_ = this->layer_param_.downsample_param().top_height();
    top_width_ = this->layer_param_.downsample_param().top_width();
  } else {
    top_height_ = bottom[1]->height();
    top_width_ = bottom[1]->width();
  }
  
  CHECK_GE(top_height_, 1) << "DownsampleLayer must have top_height > 0";
  CHECK_GE(top_width_, 1) << "DownsampleLayer must have top_width > 0";
  
  height_ = top_height_;
  width_ = top_width_;
  
  count_ = width_ * height_ * channels_ * num_;
  
  top[0]->Reshape(num_, channels_, height_, width_);
  CHECK_EQ(count_, top[0]->count());
  
  if(bottom[0]->width() == width_ && bottom[0]->height() == height_) {   
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]); 
  }
}

template <typename Dtype>
void DownsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void DownsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for(int i=0; i<propagate_down.size(); i++)
        if(propagate_down[i])
                LOG(FATAL) << "DownsamplingLayer cannot do backward.";
}

#ifdef CPU_ONLY
STUB_GPU(DownsampleLayer);
#endif

INSTANTIATE_CLASS(DownsampleLayer);
REGISTER_LAYER_CLASS(Downsample);

}  // namespace caffe

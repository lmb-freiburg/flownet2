#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/correlation_layer.hpp"

namespace caffe {

template <typename Dtype>
void CorrelationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  CorrelationParameter corr_param = this->layer_param_.correlation_param();
  
  CHECK(corr_param.has_kernel_size()) << "Filter kernel_size is not set";
  CHECK(corr_param.has_max_displacement()) << "Max displacement is required.";
  
  kernel_size_ = corr_param.kernel_size();
  if(kernel_size_ % 2 == 0) LOG(FATAL) << "Odd kernel size required";
  
  max_displacement_ = corr_param.max_displacement();
  pad_size_ = corr_param.pad();
  stride1_ = corr_param.stride_1();
  stride2_ = corr_param.stride_2();
  
  do_abs_ = corr_param.do_abs();
  
  corr_type_ = corr_param.correlation_type();
  
  LOG(INFO) << "Kernel Size: " << kernel_size_;
  LOG(INFO) << "Stride 1: " << stride1_;
  LOG(INFO) << "Stride 2: " << stride2_;
  LOG(INFO) << "Max Displacement: " << max_displacement_;
  
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  num_ = bottom[0]->num();
  
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "Both bottom blobs must have same width";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "Both bottom blobs must have same height";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "Both bottom blobs must have same height";

  int bottomchannels = bottom[0]->channels();
  
  int paddedbottomheight = bottom[0]->height()+2*pad_size_;
  int paddedbottomwidth = bottom[0]->width()+2*pad_size_;  
  
  // Size computation
  kernel_radius_ = (kernel_size_ - 1) / 2; //size of unreachable border region (on each side)
  border_size_ = max_displacement_ + kernel_radius_; //size of unreachable border region (on each side)
  
  top_width_ = ceil((float)(paddedbottomwidth - border_size_*2) / (float)stride1_);
  top_height_ = ceil((float)(paddedbottomheight - border_size_*2) / (float)stride1_);

  CHECK_GE(top_width_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";
  CHECK_GE(top_height_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";
  
  // Given a center position in image 1, how many displaced positions in -x / +x direction do we consider in image 2 (neighborhoodGridWidth):
  neighborhood_grid_radius_ = max_displacement_ / stride2_;
  neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

  // Top Channels amount to displacement combinations in X and Y direction:
  top_channels_ = neighborhood_grid_width_ * neighborhood_grid_width_;
  
  //Reshape top
  top[0]->Reshape(num_, top_channels_, top_height_, top_width_);

  // rbots (These are the blobs that store the padded and dimension rearranged data
  rbot1_.reset(new Blob<Dtype>());
  rbot2_.reset(new Blob<Dtype>());
  rbot1_->Reshape(num_, paddedbottomheight, paddedbottomwidth, bottomchannels);
  rbot2_->Reshape(num_, paddedbottomheight, paddedbottomwidth, bottomchannels);
  
  rtopdiff_.reset(new Blob<Dtype>());
  rtopdiff_->Reshape(num_, top_height_, top_width_, top_channels_);

}

template <typename Dtype>
void CorrelationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CorrelationLayer);
#endif

INSTANTIATE_CLASS(CorrelationLayer);
REGISTER_LAYER_CLASS(Correlation);

}  // namespace caffe

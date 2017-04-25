#ifndef FLOW_AUGMENTATION_LAYER_HPP_
#define FLOW_AUGMENTATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/augmentation_layer_base.hpp"

namespace caffe {

/**
 * @brief Optical Flow Augmentation Layer
 *
 */
template <typename Dtype>
class FlowAugmentationLayer : public AugmentationLayerBase<Dtype>, public Layer<Dtype> {
 public:
  explicit FlowAugmentationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~FlowAugmentationLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool AllowBackward() const { LOG(WARNING) << "FlowAugmentationLayer does not do backward."; return false; }
  

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "FlowAugmentationLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "FlowAugmentationLayer cannot do backward."; return; }
      

  shared_ptr<SyncedMemory> coeff_matrices1_;
  shared_ptr<SyncedMemory> coeff_matrices2_;
  Blob<Dtype> all_coeffs1_;
  Blob<Dtype> all_coeffs2_;
  
  Blob<Dtype> test_coeffs_;
  
  int cropped_height_;
  int cropped_width_;
  int num_params_;
};


}  // namespace caffe

#endif  // FLOW_AUGMENTATION_LAYER_HPP_

#ifndef GENERATE_AUGMENTATION_PARAMETERS_LAYER_HPP_
#define GENERATE_AUGMENTATION_PARAMETERS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/augmentation_layer_base.hpp"

namespace caffe {

/**
 * @brief Generate Augmentation Parameters Layer
 *
 */
template <typename Dtype>
class GenerateAugmentationParametersLayer : public AugmentationLayerBase<Dtype>, public Layer<Dtype> {
 public:
  explicit GenerateAugmentationParametersLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~GenerateAugmentationParametersLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool AllowBackward() const { LOG(WARNING) << "GenerateAugmentationParametersLayer does not do backward."; return false; }
  
 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "GenerateAugmentationParametersLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "GenerateAugmentationParametersLayer cannot do backward."; return; }
      
    
  int num_params_; 
  int num_iter_;
  int cropped_width_;
  int cropped_height_;
  int bottomwidth_;
  int bottomheight_;
  int num_;
  AugmentationParameter aug_;
  CoeffScheduleParameter discount_coeff_schedule_;
  std::string mode_;
};


}  // namespace caffe

#endif  // GENERATE_AUGMENTATION_PARAMETERS_LAYER_HPP_

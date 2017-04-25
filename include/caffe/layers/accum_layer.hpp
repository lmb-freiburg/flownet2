#ifndef ACCUM_LAYER_HPP_
#define ACCUM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief AccumLayer
 *        Takes at least two blobs and concatenates them along channels
 *        but upscales smaller blobs in width and height to match the largest
 */
template <typename Dtype>
class AccumLayer : public Layer<Dtype> {
 public:
  explicit AccumLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "Accum"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //Blob<Dtype> col_bob_;
  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int concat_dim_;
  
  int top_width_;
  int top_height_;
};

}  // namespace caffe

#endif  // ACCUM_LAYER_HPP_

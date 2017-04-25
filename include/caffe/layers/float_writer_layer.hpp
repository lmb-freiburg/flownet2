#ifndef FLOAT_WRITER_LAYER_HPP_
#define FLOAT_WRITER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief FloatWriterLayer writes float3 files
 *
 */
template <typename Dtype>
class FloatWriterLayer : public Layer<Dtype> {
 public:
  explicit FloatWriterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~FloatWriterLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "FloatWriter"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline bool AllowInactive() const { return false; }
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}  // namespace caffe

#endif  // FLOAT_WRITER_LAYER_HPP_

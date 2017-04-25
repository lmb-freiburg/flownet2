#ifndef PFM_WRITER_LAYER_HPP_
#define PFM_WRITER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief PFMWriterLayer writes PFM files
 *
 */
template <typename Dtype>
class PFMWriterLayer : public Layer<Dtype> {
 public:
  explicit PFMWriterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~PFMWriterLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "PFMWriter"; }
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

#endif  // PFM_WRITER_LAYER_HPP_

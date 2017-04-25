#ifndef IMG_READER_LAYER_HPP_
#define IMG_READER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ImgReaderLayer : public Layer<Dtype> {
 public:
  explicit ImgReaderLayer(const LayerParameter& param)
      : Layer<Dtype>(param) { }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  Blob<Dtype> data_;
  void ReadData();

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "FloatReaderLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "FloatReaderLayer cannot do backward."; return; }
};

}  // namespace caffe

#endif  // IMG_READER_LAYER_HPP_

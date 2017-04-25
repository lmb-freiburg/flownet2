#ifndef FLOAT_READER_LAYER_HPP_
#define FLOAT_READER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class FloatReaderLayer : public Layer<Dtype> {
 public:
  explicit FloatReaderLayer(const LayerParameter& param)
      : Layer<Dtype>(param) { data_=0; dataXSize_=0; dataYSize_=0; dataZSize_=0; }
  virtual ~FloatReaderLayer() { if(data_) delete data_; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  int dataXSize_;
  int dataYSize_;
  int dataZSize_;
  float* data_;
  void ReadData();

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "FloatReaderLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "FloatReaderLayer cannot do backward."; return; }
};

}  // namespace caffe

#endif

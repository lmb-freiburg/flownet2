#ifndef LPQ_LOSS_LAYER_HPP_
#define LPQ_LOSS_LAYER_HPP_

#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

/**
 * L_p,q loss: Loss function f(x) := (x^p)^q
 * E.g.: L1 is L_1,1 
 *       squared L2 is L_2,1 
 *       Euclidean is L_2,2
 */

template <typename Dtype>
class LpqLossLayer : public LossLayer<Dtype> {
 public:
  explicit LpqLossLayer(
        const LayerParameter& param)
      : LossLayer<Dtype>(param), sign_() {}
  virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);    
  virtual void Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "L1Loss"; }
  
  virtual inline bool AllowForceBackward(
        const int bottom_index) const 
  {
    return true;
  }
  
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  
 protected:
  /// @copydoc LpqLossLayer
  virtual void Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(
        const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(
        const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom);

  /**
   * See "LpqLossParameter" in caffe.proto
   */
  struct ScheduleStep_
  {
    ScheduleStep_(unsigned int i, Dtype p, Dtype q)
      : start_iter(i), p(p), q(q) {}
    
    unsigned int start_iter;
    Dtype p;
    Dtype q;
  };
  std::queue<ScheduleStep_*> schedule_;
  
  Blob<Dtype> sign_, mask_, plateau_l2_, ones_;
  float scale_;
  Dtype normalize_coeff_;
  
  // Extra layers to do the dirty work using already implemented stuff
  shared_ptr<EltwiseLayer<Dtype> > diff_layer_;
  Blob<Dtype> diff_;
  vector<Blob<Dtype>*> diff_top_vec_;
  
  shared_ptr<PowerLayer<Dtype> > p_layer_;
  Blob<Dtype> p_output_;
  vector<Blob<Dtype>*> p_top_vec_;
  
  shared_ptr<ConvolutionLayer<Dtype> > sum_layer_;
  Blob<Dtype> sum_output_;
  vector<Blob<Dtype>*> sum_top_vec_;
  
  shared_ptr<PowerLayer<Dtype> > q_layer_;
  Blob<Dtype> q_output_;
  vector<Blob<Dtype>*> q_top_vec_;
};


}  // namespace caffe

#endif  // LPQ_LOSS_LAYER_HPP_

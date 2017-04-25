#include <iomanip>
#include <sstream>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lpq_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

  template <typename Dtype>
  void LpqLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) 
  {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    
    /// P/Q parameters schedule
    {
      const google::protobuf::RepeatedField<unsigned int>& start_iters =
        this->layer_param_.lpq_loss_param().pq_episode_starts_at_iter();
      const google::protobuf::RepeatedField<float>& ps =
        this->layer_param_.lpq_loss_param().p();
      const google::protobuf::RepeatedField<float>& qs =
        this->layer_param_.lpq_loss_param().q();
      /// Special case: If one of each p and q is given, it's okay to not 
      /// specify a start frame
      if (start_iters.size() == 0 and
          ps.size() == 1 and
          qs.size() == 1)
      {
        schedule_.push(new ScheduleStep_(0, ps.Get(0), qs.Get(0)));
        LOG(INFO) << "Lpq loss layer: Constant parameters"
                  << " p = "  << schedule_.front()->p
                  << ", q = " << schedule_.front()->q;
      }
      else
      {
        /// Ensure that all three vectors are the same size...
        if (start_iters.size() != ps.size() or  ps.size() != qs.size()) {
          LOG(FATAL) << "Incompatible sizes: ("
                    << "pq_episode_starts_at_iter -> " 
                    << start_iters.size() << ", "
                    << "p -> " << ps.size() << ", "
                    << "q -> " << qs.size() << ")";
        }
        /// ...and are not empty
        if (start_iters.size() < 1) {
          LOG(FATAL) << "Lpq schedule parameters must not be empty";
        }
        /// The start frames must also be ordered and not contain duplicates
        if (start_iters.size() > 1) {
          for (unsigned int i = 0; i < start_iters.size()-1; ++i) {
            if (start_iters.Get(i) >= start_iters.Get(i+1)) {
              std::ostringstream oss;
              for (unsigned int j = 0; j < start_iters.size(); ++j) {
                oss << " " << start_iters.Get(j);
              }
              LOG(FATAL) << "pq_episode_starts_at_frame is not ordered "
                         << "or contains duplicates:" << oss.str();
            }
          }
        }
        /// Also it is probably a mistake if the first start frame is NOT zero
        if (start_iters.Get(0) != 0) {
          LOG(FATAL) << "First entry in pq_episode_starts_at_frame is not 0, "
                     << "this is probably a mistake.";
        }
        /// Make schedule
        for (unsigned int i = 0; i < start_iters.size(); ++i) {
          schedule_.push(new ScheduleStep_(start_iters.Get(i), 
                                           ps.Get(i), 
                                           qs.Get(i)));
          LOG(INFO) << "Lpq loss layer: Starting at iteration "
                    << std::setw(7) << std::setfill(' ')
                    << schedule_.back()->start_iter
                    << ": p = " << schedule_.back()->p
                    << ", q = " << schedule_.back()->q;
        }
      }
    }
    
    if(bottom.size() == 1) {
      diff_top_vec_.clear();
      diff_top_vec_.push_back(bottom[0]);
    } else if(bottom.size() == 2) {
      // Set up eltwise layer to compute elementwise difference
      diff_top_vec_.clear();
      diff_top_vec_.push_back(&diff_);
      LayerParameter diff_param;
      diff_param.mutable_eltwise_param()->add_coeff(1.);
      diff_param.mutable_eltwise_param()->add_coeff(-1.);
      diff_param.mutable_eltwise_param()->set_operation(
                                            EltwiseParameter_EltwiseOp_SUM);
      diff_layer_.reset(new EltwiseLayer<Dtype>(diff_param));
      diff_layer_->SetUp(bottom, diff_top_vec_);
    } else {
      LOG(FATAL) << "LpqLossLayer needs one or two input blobs.";
    }
    
    /// Set up power layer to compute elementwise p-power
    p_top_vec_.clear();
    p_top_vec_.push_back(&p_output_);
    LayerParameter p_param;
    p_param.mutable_power_param()->set_power(schedule_.front()->p);
    p_param.mutable_power_param()->set_shift(
        this->layer_param_.lpq_loss_param().p_epsilon());
    p_layer_.reset(new PowerLayer<Dtype>(p_param));
    p_layer_->SetUp(diff_top_vec_, p_top_vec_);
    
    /// Set up convolutional layer to sum all channels
    sum_top_vec_.clear();
    sum_top_vec_.push_back(&sum_output_);
    LayerParameter sum_param;  
    sum_param.mutable_convolution_param()->set_num_output(1);
    sum_param.mutable_convolution_param()->add_kernel_size(1);
    sum_param.mutable_convolution_param()->mutable_weight_filler()
                                         ->set_type("constant");
    if(this->layer_param_.lpq_loss_param().l2_prescale_by_channels()) {
      sum_param.mutable_convolution_param()->mutable_weight_filler()
                                           ->set_value(Dtype(1)/
                                                       Dtype(bottom[0]->channels()));
    } else {
      sum_param.mutable_convolution_param()->mutable_weight_filler()
                                           ->set_value(Dtype(1));
    }
    sum_layer_.reset(new ConvolutionLayer<Dtype>(sum_param));
    sum_layer_->SetUp(p_top_vec_, sum_top_vec_);
    
    /// Set up power layer to compute elementwise q-power
    q_top_vec_.clear();
    q_top_vec_.push_back(&q_output_);
    LayerParameter q_param;
    q_param.mutable_power_param()->set_power(schedule_.front()->q);
    q_param.mutable_power_param()->set_shift(
        this->layer_param_.lpq_loss_param().q_epsilon());
    q_layer_.reset(new PowerLayer<Dtype>(q_param));
    q_layer_->SetUp(sum_top_vec_, q_top_vec_);
    
    /// Discard first parameter schedule step
    schedule_.pop();
  }

  template <typename Dtype>
  void LpqLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                    const vector<Blob<Dtype>*>& top) 
  {
    vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
    top[0]->Reshape(loss_shape);
    
    if(bottom.size() > 1) {
      diff_layer_->Reshape(bottom, diff_top_vec_);
    }
    
    sign_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                  bottom[0]->height(), bottom[0]->width());
  
    mask_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                  bottom[0]->height(), bottom[0]->width());

//     plateau_l2_.ReshapeLike(sum_output_);
    
    ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                  bottom[0]->height(), bottom[0]->width());
    caffe_set(ones_.count()/ones_.channels(), Dtype(1), ones_.mutable_cpu_data());
    
    p_layer_->Reshape(diff_top_vec_, p_top_vec_);
    sum_layer_->Reshape(p_top_vec_, sum_top_vec_);
    q_layer_->Reshape(sum_top_vec_, q_top_vec_);    
    
    //caffe_set(sign_.count()/sign_.channels(), Dtype(1), sign_.mutable_cpu_data());
  }

  template <typename Dtype>
  void LpqLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) 
  {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void LpqLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, 
                                        const vector<Blob<Dtype>*>& bottom) 
  {
    NOT_IMPLEMENTED;
  }

  #ifdef CPU_ONLY
  STUB_GPU(LpqLossLayer);
  #endif

  INSTANTIATE_CLASS(LpqLossLayer);
  REGISTER_LAYER_CLASS(LpqLoss);

}  // namespace caffe

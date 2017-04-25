#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lpq_loss_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {
  
  namespace LpqLayer__kernels {

    /**
     * @brief Elementwise sign
     * @param n Number of data elements
     * @param in Input data
     * @param out Output data
     */
    template <typename Dtype>
    __global__ void ComputeSign(const int n, 
                                const Dtype* in, 
                                Dtype* out) 
    {
      CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index] > 0 ? Dtype(1) : Dtype(-1);
      }
    } 

    // TODO maybe change the way of detecting NaNs

    template <typename Dtype>
    __global__ void FindNotNaNs(const int n, 
                                const Dtype* in, 
                                Dtype* out) 
    {
      CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index]==in[index] ? Dtype(1) : Dtype(0);
      }
    } 

    template <typename Dtype>
    __global__ void KillNaNs(const int n, 
                             const Dtype* in, 
                             Dtype* out) 
    {
      CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index]==in[index] ? in[index] : Dtype(0);
      }
    }

    template <typename Dtype>
    __global__ void KillMasked(const int n, 
                               const Dtype* in, 
                               Dtype* out) 
    {
      CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index] > Dtype(0.5) ? out[index] : Dtype(0);
      }
    }

    template <typename Dtype>
    __global__ void KillMaskedAcrossChannels(const int n, 
                                             const int width_height, 
                                             const Dtype* in, 
                                             Dtype* out) 
    {
      CUDA_KERNEL_LOOP(index, n) {
        const int mask_idx = index % width_height;
        out[index] = in[mask_idx] > Dtype(0.5) ? out[index] : Dtype(0);
      }
    }

    /**
     * @brief Elementwise multiplication
     * @param n Number of data points
     * @param data Input/output data
     * @param factors Factors to be multiplied onto "data" (same size)
     */
    template <typename Dtype>
    __global__ void EltwiseMult(const int n, 
                                Dtype* data, 
                                const Dtype* factors)
    {
      CUDA_KERNEL_LOOP(index, n) {
        data[index] *= factors[index];
      }
    }

  }  // namespace LpqLayer__kernels


  /**
   * Forward propagation
   */
  template <typename Dtype>
  void LpqLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
  {
    /// Check and reset p/q parameters
    if (schedule_.size() > 0) {
      /// Get current iteration
      Net<Dtype> *net = this->GetNet();
      unsigned int current_iteration = (net) ? net->iter() : 0;
      
      ScheduleStep_* step_ptr = 0;
      
      /// Discard old schedule steps
      while (schedule_.size() > 0 and 
             current_iteration >= schedule_.front()->start_iter)
      {
        if (step_ptr) {
          delete step_ptr;
        }
        step_ptr = schedule_.front();
        LOG(INFO) << "Lpq loss layer: Pop'ing schedule step: "
                  << "start_iter = " << step_ptr->start_iter
                  << ", p = " << step_ptr->p
                  << ", q = " << step_ptr->q;
        schedule_.pop();
      }
      
      /// Use retrieved schedule step and reinitialize p/q layers
      if (step_ptr) {
        LOG(INFO) << "Lpq loss layer: Iteration " << current_iteration
                  << ", switching to p = " << step_ptr->p
                  << ", q = " << step_ptr->q;
        
        p_layer_->SetPower(step_ptr->p);
        q_layer_->SetPower(step_ptr->q);
        
        /// Discard used schedule step
        delete step_ptr;
      }
    }
    /// <-- Check and reset p/q parameters
    
    
    Blob<Dtype> *diffptr = diff_top_vec_[0];
    
    Dtype dot, loss;
    if(bottom.size() > 1) {
      diff_layer_->Forward(bottom, diff_top_vec_);
    }
    
    // if necessary, compute the number of not-NaNs
    int count = bottom[0]->count();
    int num   = bottom[0]->num();
    LpqLayer__kernels::FindNotNaNs<Dtype>
                                <<<CAFFE_GET_BLOCKS(count), 
                                   CAFFE_CUDA_NUM_THREADS>>>(
                                      count, 
                                      diffptr->gpu_data(), 
                                      mask_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    
    if (this->layer_param_.lpq_loss_param().normalize_by_num_entries()) {    
      caffe_gpu_dot(count, mask_.gpu_data(), mask_.gpu_data(), &normalize_coeff_);
      normalize_coeff_ /= mask_.channels();
    } else {
      normalize_coeff_ = num;
    }
    
    /// set masked (NaNs only) to zero
    LpqLayer__kernels::KillMasked<Dtype><<<CAFFE_GET_BLOCKS(count),
                                           CAFFE_CUDA_NUM_THREADS>>>(
                                              count, 
                                              mask_.gpu_data(), 
                                              diffptr->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    
    /// Compute sign of data
    LpqLayer__kernels::ComputeSign<Dtype><<<CAFFE_GET_BLOCKS(count),
                                            CAFFE_CUDA_NUM_THREADS>>>(
                                              count, 
                                              diffptr->gpu_data(),
                                              sign_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    
    /// Convert data to absolute (elementwise product with sign)
    LpqLayer__kernels::EltwiseMult<Dtype>
                                <<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
                                      count, 
                                      diffptr->mutable_gpu_data(),
                                      sign_.gpu_data());
    CUDA_POST_KERNEL_CHECK;
    
    p_layer_->Forward(diff_top_vec_, p_top_vec_);
    sum_layer_->Forward(p_top_vec_, sum_top_vec_);
    q_layer_->Forward(sum_top_vec_, q_top_vec_);
    
    /// Sum up all loss values
    caffe_gpu_dot(q_output_.count(), q_output_.gpu_data(), ones_.gpu_data(), &dot);
      
    loss = dot / normalize_coeff_; 
    top[0]->mutable_cpu_data()[0] = loss;
  }

  
  /**
   * Backward propagation
   */
  template <typename Dtype>
  void LpqLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down, 
                                         const vector<Blob<Dtype>*>& bottom)
  {
    bool prop_down = propagate_down[0];
    if(bottom.size() > 1) prop_down |= propagate_down[1];
    
    Blob<Dtype> *diffptr = diff_top_vec_[0];
    
    if (prop_down) {
      const Dtype alpha = top[0]->cpu_diff()[0] / normalize_coeff_;
    
      vector<bool> prop_down(1,true);
      caffe_set(q_output_.count(), alpha, q_output_.mutable_cpu_diff());

      q_layer_->Backward(q_top_vec_, prop_down, sum_top_vec_);
      sum_layer_->Backward(sum_top_vec_, prop_down, p_top_vec_);
      p_layer_->Backward(p_top_vec_, prop_down, diff_top_vec_);
      
      /// Restore original sign of data
      LpqLayer__kernels::EltwiseMult<Dtype>
                                <<<CAFFE_GET_BLOCKS(diffptr->count()),
                                   CAFFE_CUDA_NUM_THREADS>>>(
                                      diffptr->count(),
                                      diffptr->mutable_gpu_diff(),
                                      sign_.gpu_data());
      CUDA_POST_KERNEL_CHECK;
      
      LpqLayer__kernels::KillMasked<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()),
                                             CAFFE_CUDA_NUM_THREADS>>>(
                                                diffptr->count(), 
                                                mask_.gpu_data(), 
                                                diffptr->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      
      if(bottom.size() > 1) {
        diff_layer_->Backward(diff_top_vec_, propagate_down, bottom);
      }
    }
    
  }

  INSTANTIATE_LAYER_GPU_FUNCS(LpqLossLayer);

}  // namespace caffe

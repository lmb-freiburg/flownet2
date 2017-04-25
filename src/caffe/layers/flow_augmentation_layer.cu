// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/flow_augmentation_layer.hpp"
#include "caffe/layers/data_augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <fstream>

#define CUDART_NAN_F            __int_as_float(0x7fffffff)

namespace caffe {

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}


template <typename Dtype> 
__global__ void WarpData(const int nthreads, const int num, const int height, const int width, const Dtype* src_data, const int src_count,
                                const int dest_height, const int dest_width, Dtype* dest_data,
                                const typename AugmentationLayerBase<Dtype>::tTransMat *transMats1,
                                const typename AugmentationLayerBase<Dtype>::tTransMat *transMats2
                        ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    float x = (float)(index % dest_width); //w-pos
    float y = (float)((index / dest_width) % dest_height); //h-pos
    int n = (index / dest_width / dest_height); // num
    
    // === Warping:
    //transMat:
    // / 0 2 4 \
    // \ 1 3 5 /
    const typename DataAugmentationLayer<Dtype>::tTransMat *transMat1 = &(transMats1[n]);
    const typename DataAugmentationLayer<Dtype>::tTransMat *transMat2 = &(transMats2[n]);
    
    float xpos1, ypos1, xpos2, ypos2, xpos3, ypos3;
    // Step 1: Apply inverse tranformation of Image 1
    xpos1 = x * transMat1->t0 + y * transMat1->t2 + transMat1->t4;
    ypos1 = x * transMat1->t1 + y * transMat1->t3 + transMat1->t5;

    // Step 2: Apply flow field
    int srcIdxOffx = width*(height*(2*n+0) + 
                     (int)(ypos1+(Dtype)0.5)) + 
                     (int)(xpos1+(Dtype)0.5);
    int srcIdxOffy = width*(height*(2*n+1) + 
                     (int)(ypos1+(Dtype)0.5)) + 
                     (int)(xpos1+(Dtype)0.5);
    
    xpos2 = xpos1 + src_data[min(srcIdxOffx,src_count)];
    ypos2 = ypos1 + src_data[min(srcIdxOffy,src_count)];
    
    // Step 3: Apply tranformation of Image 2
    xpos3 = xpos2 * transMat2->t0 + ypos2 * transMat2->t2 + transMat2->t4;
    ypos3 = xpos2 * transMat2->t1 + ypos2 * transMat2->t3 + transMat2->t5;
    
    // Step 4: Difference between the new and old positions gives the flow
    dest_data[dest_width*(dest_height*(2*n+0) + (int)y) + (int)x] = xpos3 - x;
    dest_data[dest_width*(dest_height*(2*n+1) + (int)y) + (int)x] = ypos3 - y;
    
    /*xpos = clamp(xpos, 0.0f, (float)(width)-1.05f);//Ensure that floor(xpos)+1 is still valid
    ypos = clamp(ypos, 0.0f, (float)(height)-1.05f);
    
    // Get interpolated sample
    //float sample = tex2DLayered(texRef, xpos, ypos, cn);
    float tlx = floor(xpos);
    float tly = floor(ypos);
    int srcIdxOff = width*(height*cn + tly) + tlx;
    
    float sampleTL = src_data[srcIdxOff];
    float sampleTR = src_data[min(srcIdxOff+1,src_count)];
    float sampleBL = src_data[min(srcIdxOff+width,src_count)];
    float sampleBR = src_data[min(srcIdxOff+1+width,src_count)];
    
    float xdist = xpos - tlx;
    float ydist = ypos - tly;
    
    float sample = (1-xdist)*(1-ydist)*sampleTL
                 + (  xdist)*(  ydist)*sampleBR
                 + (1-xdist)*(  ydist)*sampleBL
                 + (  xdist)*(1-ydist)*sampleTR;
    
    dest_data[index] = sample;*/
  }
}


template <typename Dtype>
void FlowAugmentationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  //"Flow augmentation layer takes three input blobs: FlowField, Img1TransfParams, Img2TransfParams";
  
  Dtype* top_data = (top)[0]->mutable_gpu_data(); // dest
  int topwidth = (top)[0]->width();
  int topheight = (top)[0]->height();
  int topchannels = (top)[0]->channels();
  int topcount = (top)[0]->count();
  
  CHECK_EQ(topchannels, 2);
  
  
  const Dtype* bottom_data = bottom[0]->gpu_data(); // source
  int bottomchannels = (bottom)[0]->channels();
  int bottomwidth = (bottom)[0]->width();
  int bottomheight = (bottom)[0]->height();
  int bottomcount = (bottom)[0]->count();
  
  int num = (bottom)[0]->num();
  
  CHECK_EQ(bottomchannels, 2);
  CHECK_EQ((bottom)[0]->num(), (top)[0]->num());
  
  // Debug: check for NaNs and lare values:    
  const Dtype* bottom_cpu_data = bottom[0]->cpu_data();
  /*for(int i=0; i<bottomcount; i++) {
      if (isnan(bottom_cpu_data[i]))
          LOG(WARNING) << "bottom_data[" << i << "]=NaN";
//       if (std::fabs(bottom_cpu_data[i])>1e3)
//           LOG(WARNING) << "bottom_data[" << i << "]=" << bottom_cpu_data[i];
  }*/
  
  
  // Prepare matrices
  all_coeffs1_.ShareData(*bottom[1]); //reuse
  all_coeffs2_.ShareData(*bottom[2]);
  //all_coeffs1_.CopyFrom(*bottom[1]);
  //all_coeffs2_.CopyFrom(*bottom[2]);
  const Dtype* my_params1 = all_coeffs1_.cpu_data();
  const Dtype* my_params2 = all_coeffs2_.cpu_data();
  
  typename AugmentationLayerBase<Dtype>::tTransMat *matrices1 = (typename AugmentationLayerBase<Dtype>::tTransMat *)(coeff_matrices1_->mutable_cpu_data());
  typename AugmentationLayerBase<Dtype>::tTransMat *matrices2 = (typename AugmentationLayerBase<Dtype>::tTransMat *)(coeff_matrices2_->mutable_cpu_data());
  
  for (int item_id = 0; item_id < num; ++item_id) {
    AugmentationCoeff coeff; 

    // Load the previously generated coeffs (either they are from another layer or generated above)
    AugmentationLayerBase<Dtype>::array_to_coeff(my_params1 + item_id * num_params_, coeff);

    matrices1[item_id].toIdentity();
    matrices1[item_id].fromCoeff(&coeff,cropped_width_,cropped_height_,bottomwidth,bottomheight);

    AugmentationLayerBase<Dtype>::array_to_coeff(my_params2 + item_id * num_params_, coeff);
    matrices2[item_id].toIdentity();
    matrices2[item_id].fromCoeff(&coeff,cropped_width_,cropped_height_,bottomwidth,bottomheight);
    matrices2[item_id] = matrices2[item_id].inverse();
  }
  

  // Do GPU work  
  typename AugmentationLayerBase<Dtype>::tTransMat *gpumatrices1 = (typename AugmentationLayerBase<Dtype>::tTransMat *)(coeff_matrices1_->gpu_data());
  typename AugmentationLayerBase<Dtype>::tTransMat *gpumatrices2 = (typename AugmentationLayerBase<Dtype>::tTransMat *)(coeff_matrices2_->gpu_data());
  
  int topThreadCount = topcount / 2;
  WarpData<Dtype><<<CAFFE_GET_BLOCKS(topThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
      topThreadCount,
      num, bottomheight, bottomwidth, bottom_data, bottomcount,
      topheight, topwidth, top_data, gpumatrices1, gpumatrices2);

  CUDA_POST_KERNEL_CHECK;
  
}

INSTANTIATE_LAYER_GPU_FUNCS(FlowAugmentationLayer);

}  // namespace caffe

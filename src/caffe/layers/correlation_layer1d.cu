#include <vector>
#include <stdio.h>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output.hpp"

#include "caffe/layers/correlation_1d_layer.hpp"

#include "caffe/caffe.hpp"

#define ROUND_OFF 50000

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

namespace caffe {

// == Dimension rearrangement Kernel
  
namespace corr1d {
  
template <typename Dtype>
__global__ void blob_rearrange_kernel2(const Dtype* in, Dtype* out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;

    float value=in[(n*channels+ch)*widthheight+xy];

    __syncthreads();

    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + 0);
    int xypad = ypad * (width+2*padding) + xpad;

    out[(n*pwidthheight+xypad)*channels + ch] = value;
}

// == Correlation Kernel
template <typename Dtype> 
__global__ void CorrelateData(const int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *top) 
{
  extern __shared__ char patch_data_char[];
  
  Dtype *patch_data = (Dtype *)patch_data_char;
  
    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
  int x1 = blockIdx.x*stride1 + max_displacement;
  int y1 = blockIdx.y*stride1;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;
  
  // Load 3D patch into shared shared memory
  for(int j = 0; j < kernel_size; j++) { // HEIGHT
    for(int i = 0; i < kernel_size; i++) { // WIDTH
      int ji_off = ((j * kernel_size) + i) * bottomchannels;
      for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = bottom0[idx1];
      }
    }
  }
  
  __syncthreads();
  
  __shared__ Dtype sum[WARPS_PER_BLOCK*THREADS_PER_WARP];
  
  // Compute correlation
  for(int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;
  
    int s2o = (top_channel % neighborhood_grid_width + x_shift) * stride2;
    
    for(int j = 0; j < kernel_size; j++) { // HEIGHT
      for(int i = 0; i < kernel_size; i++) { // WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
        for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int x2 = x1 + s2o;
          
          int idxPatchData = ji_off + ch;
          int idx2 = ((item * bottomheight + y1+j) * bottomwidth + x2+i) * bottomchannels + ch;
          
          sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
        }
      }
    }
    
    __syncthreads();
    
    if(ch_off == 0) {
        Dtype total_sum = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
            total_sum += sum[idx];
        }
        const int sumelems = kernel_size*kernel_size*bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
    }
  }
  
  
  // Aggregate  
}

// == Correlation Backward Pass Kernel (For Blob 0)
template <typename Dtype> 
__global__ void CorrelateDataBackward0(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  Dtype *bottom0diff, const Dtype *bottom1, const Dtype *topdiff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight; //h-pos

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    
    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
    

    Dtype sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        {
          for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int idxbot1 = ((item * pbottomheight + m) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            Dtype bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m,n]

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot1tmp;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
		const int bot0index = ((n * bottomheight) + m) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
  }

}



// == Correlation Backward Pass Kernel (For Blob 1)
template <typename Dtype> 
__global__ void CorrelateDataBackward1(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const Dtype *bottom0, Dtype *bottom1diff, const Dtype *topdiff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    //int l = index % bottomwidth + pad_size; //w-pos
    //int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
    //int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight; //h-pos
    
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    Dtype sum = 0;
    {
      
      for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {
        
        int s2o = stride2 * o;
        
        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        
        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - 0 - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - 0) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + m) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            Dtype bot0tmp = bottom0[idxbot0]; // bottom1[l+s2o,m,n]

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot0tmp;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
		const int bot1index = ((n * bottomheight) + m) * bottomwidth + (l-pad_size);
		bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
  }

}

// == Correlation Kernel Subtraction
template <typename Dtype> 
__global__ void CorrelateDataSubtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *top) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x = index % topwidth; //w-pos
    int y = (index / topwidth) % topheight; //h-pos
    int c = (index / topwidth / topheight) % topchannels; //channels
        
    // Offset of patch in image 2
    int s2o = (c % neighborhood_grid_width + x_shift) * stride2;
        
    // First (upper left) position of kernel center in current neighborhood in image 1
    int x1 = x*stride1 + kernel_radius + max_displacement;
    int y1 = y*stride1 + kernel_radius + 0;
    
    // Iterate through 3D patch
    Dtype sum = 0;
    for(int j = -kernel_radius; j <= kernel_radius; j++) { // HEIGHT
      for(int i = -kernel_radius; i <= kernel_radius; i++) { // WIDTH
        for(int l = 0; l < bottomchannels; l++) { // CHANNELS
          // Calculate position in image 2
          int x2 = x1 + s2o;
          int y2 = y1;

          // Indices in bottom data: (CH=l,W=x2,H=y2,N)
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + l;
          int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + l;

          // Do the correlation:
          sum += fabsf(bottom0[idx1] - bottom1[idx2]);
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    top[index + item*topcount] = sum / (float)sumelems;
  }

}


// == Correlation Backward Pass Kernel (For Blob 0)
template <typename Dtype> 
__global__ void CorrelateDataBackward0Subtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  Dtype *bottom0diff, const Dtype *bottom0, const Dtype *bottom1, const Dtype *topdiff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int l = index % bottomwidth + pad_size; //w-pos
    int m = (index / bottomwidth) % bottomheight; //h-pos
    int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    
    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
    

    Dtype sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        { 
          for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int idxbot = ((item * pbottomheight + (m)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            Dtype bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m,n]
            Dtype bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m,n]
            Dtype sign = (bot0tmp >= bot1tmp) ? Dtype(1.0) : Dtype(-1.0);

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    bottom0diff[index + item*bottomcount] = sum / (float)sumelems;
  }

}


// == Correlation Backward Pass Kernel (For Blob 1)
template <typename Dtype> 
__global__ void CorrelateDataBackward1Subtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *bottom1diff, const Dtype *topdiff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int l = index % bottomwidth + pad_size; //w-pos
    int m = (index / bottomwidth) % bottomheight; //h-pos
    int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    Dtype sum = 0;
    {
      for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {
        
        int s2o = stride2 * o;
        
        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        
        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - 0 - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot = ((item * pbottomheight + (m)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            Dtype bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m,n]
            Dtype bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m,n]
            Dtype sign = (bot0tmp >= bot1tmp) ? Dtype(-1.0) : Dtype(1.0);

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    bottom1diff[index + item*bottomcount] = sum / (float)sumelems;
  }

}

} // end namespace

// == Forward 

template <typename Dtype>
void Correlation1DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(bottom.size(),2);
    CHECK_EQ(top.size(),1);

    const int bnum = bottom[0]->num();
    const int bchannels = bottom[0]->channels();
    const int bheight = bottom[0]->height();
    const int bwidth = bottom[0]->width();
    const int bwidthheight = bwidth * bheight;

    const int topcount = top_width_ * top_height_ * top_channels_;
    
    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK); 
    
    cudaMemset(rbot1_->mutable_gpu_data(), 0, rbot1_->count()*sizeof(Dtype));
    cudaMemset(rbot2_->mutable_gpu_data(), 0, rbot2_->count()*sizeof(Dtype));
    
    int threads_per_block=16;
    dim3 totalBlocksRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
    const int pwidthheight = (bwidth + 2 * pad_size_) * (bheight);
    
    corr1d::blob_rearrange_kernel2<Dtype><<<totalBlocksRearr,threads_per_block>>>
            (bottom[0]->gpu_data(),rbot1_->mutable_gpu_data(),bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);
    
    corr1d::blob_rearrange_kernel2<Dtype><<<totalBlocksRearr,threads_per_block>>>
            (bottom[1]->gpu_data(),rbot2_->mutable_gpu_data(),bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);
    
    const int num = bnum;
    const int channels = bchannels;
    const int height = bheight;
    const int width = bwidth + 2*pad_size_;
    
    const int shared_memory_per_block = (kernel_size_*kernel_size_)*bchannels;

    int x_shift = - neighborhood_grid_radius_;
    if(single_direction_ == -1) { // to the left
      x_shift = -neighborhood_grid_width_;
    } else if(single_direction_ == 1) { // to the right
      x_shift = 0;
    }
    
    if(corr_type_ == CorrelationParameter_CorrelationType_MULTIPLY) {
        // Correlation1DLayer
        int topThreadCount = topcount;
        
        dim3 totalBlocksCorr(top_width_, top_height_, num);
        
        
        corr1d::CorrelateData<Dtype><<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(Dtype)>>>(
            topThreadCount,
            num, top_width_, top_height_, top_channels_, topcount,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_, kernel_size_,
            stride1_, stride2_,
            width, height, channels,
            rbot1_->gpu_data(), rbot2_->gpu_data(), top[0]->mutable_gpu_data()
            );

        CUDA_POST_KERNEL_CHECK;
        
    } else if(corr_type_ == CorrelationParameter_CorrelationType_SUBTRACT) {
        // Correlation1DLayer
        for(int n = 0; n < num; n++) {
            
            int topThreadCount = topcount;
            
            corr1d::CorrelateDataSubtract<Dtype><<<CAFFE_GET_BLOCKS(topThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
                topThreadCount,
                num, n, top_width_, top_height_, top_channels_, topcount,
                max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
                stride1_, stride2_,
                width, height, channels,
                rbot1_->gpu_data(), rbot2_->gpu_data(), top[0]->mutable_gpu_data()
                );

            
            CUDA_POST_KERNEL_CHECK;
        }
    }
}


template <typename Dtype>
void Correlation1DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{

    // Get top diff, compute bottom diff
    const Dtype* top_diff = top[0]->gpu_diff();

    Dtype* bottom0_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom1_diff = bottom[1]->mutable_gpu_diff();

    const Dtype* bottom0_data = bottom[0]->gpu_data();
    const Dtype* bottom1_data = bottom[1]->gpu_data();

    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();

    const int paddedheight = height;
    const int paddedwidth = width + 2*pad_size_;

    const int bottomcount = channels * height * width;

    int botThreadCount = bottomcount;
   
    // CorrelationLayerBackward
    
    bottom0_diff = bottom[0]->mutable_gpu_diff();
    bottom1_diff = bottom[1]->mutable_gpu_diff();
    
    int x_shift = - neighborhood_grid_radius_;
    if(single_direction_ == -1) { // to the left
      x_shift = -neighborhood_grid_width_;
    } else if(single_direction_ == 1) { // to the right
      x_shift = 0;
    }
    
    if(corr_type_ == CorrelationParameter_CorrelationType_MULTIPLY) {
        
        // == Run kernel Backward 0
        dim3 totalBlocksBackward0(width, height, channels * num); //First dim is fastest
        dim3 threadsPerBlockBackward0(THREADS_PER_WARP * WARPS_PER_BLOCK); 
        const int buffer_size_backw0 = ((int)ceil((float)(2 * kernel_radius_) / (float)stride1_) + 1) * top_channels_;
       
        // == Run kernel Backward 0 
        for(int n = 0; n < num; n++) {
        //Bottom0:
        corr1d::CorrelateDataBackward0<Dtype><<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            bottom0_diff, rbot2_->gpu_data(), top_diff
            ); 
    
        CUDA_POST_KERNEL_CHECK;
        }
        
        // == Run kernel Backward 1
        for(int n = 0; n < num; n++) {
        corr1d::CorrelateDataBackward1<Dtype><<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            rbot1_->gpu_data(), bottom1_diff, top_diff
            );
    
        CUDA_POST_KERNEL_CHECK;
        }
        
    } else if(corr_type_ == CorrelationParameter_CorrelationType_SUBTRACT) {
        for(int n = 0; n < num; n++) {
        //Bottom0:
        corr1d::CorrelateDataBackward0Subtract<Dtype><<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            bottom0_diff, rbot1_->gpu_data(), rbot2_->gpu_data(), top_diff
            );
    
        CUDA_POST_KERNEL_CHECK;
        }

        for(int n = 0; n < num; n++) {
        //Bottom0:
        corr1d::CorrelateDataBackward1Subtract<Dtype><<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            rbot1_->gpu_data(), rbot2_->gpu_data(), bottom1_diff, top_diff
            );
    
        CUDA_POST_KERNEL_CHECK;
        }
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(Correlation1DLayer);

}  // namespace caffe

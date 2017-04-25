#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/flow_warp_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp" 

#include <iostream>
#include <fstream>

#define CUDART_NAN_F            __int_as_float(0x7fffffff)

namespace caffe {

#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

//#define DISPLAY_TIMINGS

#define RA_TILE 32
#define RA_ROWS 8

template <typename Dtype>
__global__ void flow_warp_rearrange_kernel(const Dtype* in, Dtype* out, int num, int channels, int cblocks, int width, int height, int widthheight)
{
    __shared__ float buffer[RA_TILE][RA_TILE+1];

    int n  = blockIdx.x/cblocks;
    if(n>=num) return;

    int c0 = (blockIdx.x%cblocks)*RA_TILE;
    int x0 = blockIdx.y*RA_TILE;
    int y  = blockIdx.z;

    int xoff=threadIdx.x;
    int coff=threadIdx.y;
    int x=x0+xoff;

    if(x<width)
        for(int i=coff; i<RA_TILE && c0+i<channels; i+=RA_ROWS)
            buffer[i][xoff] = in[((n*channels + c0 + i)*height + y)*width + x];

    __syncthreads();

    coff = threadIdx.x;
    xoff = threadIdx.y;
    int c = c0 + coff;

    if(c<channels)
        for(int j=xoff; j<RA_TILE && x0+j<width; j+=RA_ROWS)
            out[((n*height + y)*width + x0+j)*channels + c] = buffer[coff][j];
}

#define FW_THREADS 32
#define FW_TILE_X FW_THREADS
#define FW_TILE_C FW_THREADS

template <typename Dtype>
__global__ void flow_warp_kernel_smem(const Dtype* image, const Dtype* flow, Dtype* warped, int num, int channels, int cblocks, int width, int wblocks, int height, int widthheight, float fillValue)
{
    int y = blockIdx.y;
    int n = blockIdx.z;

    __shared__ float x2_buf[FW_TILE_X], y2_buf[FW_TILE_X];
    __shared__ float buffer[FW_TILE_C][FW_TILE_X+1];

    int x;
    int c;

    x = blockIdx.x*FW_TILE_X + threadIdx.x;
    if(threadIdx.y==0 && x<width)
    {
        x2_buf[threadIdx.x] = float(x) + flow[((2*n  )*height + y)*width + x];
        y2_buf[threadIdx.x] = float(y) + flow[((2*n+1)*height + y)*width + x];
    }

    __syncthreads();

    float x2 = x2_buf[threadIdx.y];
    float y2 = y2_buf[threadIdx.y];

    int ix2_L = int(x2);
    int iy2_T = int(y2);
    int ix2_R = min(ix2_L+1, width-1);
    int iy2_B = min(iy2_T+1, height-1);

    int off_TL = ((n*height + iy2_T)*width + ix2_L)*channels;
    int off_TR = ((n*height + iy2_T)*width + ix2_R)*channels;
    int off_BL = ((n*height + iy2_B)*width + ix2_L)*channels;
    int off_BR = ((n*height + iy2_B)*width + ix2_R)*channels;

    float alpha = x2-ix2_L;
    float beta = y2-iy2_T;
    float coeffTL = (1-alpha)*(1-beta);
    float coeffTR = alpha*(1-beta);
    float coeffBL = (1-alpha)*beta;
    float coeffBR = alpha*beta;

    for(int cb=0; cb<cblocks; cb++)
    {
        __syncthreads();

        buffer[threadIdx.y][threadIdx.x] = fillValue;

        __syncthreads();

        c = cb*FW_TILE_C + threadIdx.x;
        if(x2>=0 && y2>=0 && x2<width && y2<height && c<channels)
            buffer[threadIdx.y][threadIdx.x] =  // buffer [x][c]
                coeffTL * image[off_TL + c] +
                coeffTR * image[off_TR + c] +
                coeffBL * image[off_BL + c] +
                coeffBR * image[off_BR + c];

        __syncthreads();

        c = cb*FW_TILE_C + threadIdx.y;
        x = blockIdx.x*FW_TILE_X + threadIdx.x;
        if(c<channels && x<width)
            warped[((n*channels+c)*height + y)*width + x] = buffer[threadIdx.x][threadIdx.y];
    }
}

template <typename Dtype>
__global__ void flow_warp_kernel_no_smem(const Dtype* image, const Dtype* flow, Dtype* warped, int num, int channels, int width, int wblocks, int height, int widthheight)
{
    int x = blockIdx.x*FW_TILE_X + threadIdx.x;
    if(x>=width)
        return;

    int y = blockIdx.y;
    int n = blockIdx.z;

    float x2 = float(x) + flow[((2*n  )*height + y)*width + x];
    float y2 = float(y) + flow[((2*n+1)*height + y)*width + x];

    if(x2>=0.f && y2>=0.f && x2<width && y2<height)
    {
        int ix2_L = int(x2);
        int iy2_T = int(y2);
        int ix2_R = min(ix2_L+1, width-1);
        int iy2_B = min(iy2_T+1, height-1);

        float alpha = x2-ix2_L;
        float beta = y2-iy2_T;
        for(int c=0; c<channels; c++)
        {
            int ch_off = (n*channels+c)*height;
            int off_TL = (ch_off + iy2_T)*width + ix2_L;
            int off_TR = (ch_off + iy2_T)*width + ix2_R;
            int off_BL = (ch_off + iy2_B)*width + ix2_L;
            int off_BR = (ch_off + iy2_B)*width + ix2_R;

            float coeffTL = (1-alpha)*(1-beta);
            float coeffTR = alpha*(1-beta);
            float coeffBL = (1-alpha)*beta;
            float coeffBR = alpha*beta;

            warped[(ch_off + y)*width + x] =
                coeffTL * image[off_TL] +
                coeffTR * image[off_TR] +
                coeffBL * image[off_BL] +
                coeffBR * image[off_BR];
        }
    }
}


template <typename Dtype>
__global__ void flow_warp_backward_kernel_no_smem(
        const Dtype* image_data, float* image_diff, const Dtype* flow_data, Dtype* flow_diff, const Dtype* warped_diff,
        int num, int channels, int cblocks, int width, int wblocks, int height, int widthheight)
{
    int x = blockIdx.x*FW_TILE_X + threadIdx.x;
    if(x>=width)
        return;

    int y = blockIdx.y;
    int n = blockIdx.z;

    float x2 = float(x) + flow_data[((2*n  )*height + y)*width + x];
    float y2 = float(y) + flow_data[((2*n+1)*height + y)*width + x];

    if(x2>=0.f && y2>=0.f && x2<width && y2<height)
    {
        int ix2_L = int(x2);
        int iy2_T = int(y2);
        int ix2_R = min(ix2_L+1, width-1);
        int iy2_B = min(iy2_T+1, height-1);

        float alpha=x2-ix2_L;
        float beta=y2-iy2_T;
        for(int c=0; c<channels; c++)
        {
            int ch_off = (n*channels + c)*height;
            float warped_diff_value = warped_diff[(ch_off + y)*width + x];
            atomicAdd(&image_diff[(ch_off + iy2_T)*width + ix2_L], warped_diff_value * (1-alpha)*(1-beta));
            atomicAdd(&image_diff[(ch_off + iy2_T)*width + ix2_R], warped_diff_value * alpha*(1-beta));
            atomicAdd(&image_diff[(ch_off + iy2_B)*width + ix2_L], warped_diff_value * (1-alpha)*beta);
            atomicAdd(&image_diff[(ch_off + iy2_B)*width + ix2_R], warped_diff_value * alpha*beta);
        }

        float gamma = iy2_B - y2;
        float bot_diff = 0;
        for(int c=0; c<channels; c++)
        {
            int ch_off = (n*channels + c)*height;
            float temp = 0;
            temp += gamma *     (image_data[(ch_off + iy2_T)*width + ix2_R] - image_data[(ch_off + iy2_T)*width + ix2_L]);
            temp += (1-gamma) * (image_data[(ch_off + iy2_B)*width + ix2_R] - image_data[(ch_off + iy2_B)*width + ix2_L]);

            bot_diff += warped_diff[(ch_off + y)*width + x] * temp;
        }
        flow_diff[(2*n*height + y)*width + x] = bot_diff;

        gamma = ix2_R - x2;
        bot_diff = 0;
        for(int c=0; c<channels; c++)
        {
            int ch_off = (n*channels + c)*height;
            float temp = 0;
            temp += gamma *     (image_data[(ch_off + iy2_B)*width + ix2_L] - image_data[(ch_off + iy2_T)*width + ix2_L]);
            temp += (1-gamma) * (image_data[(ch_off + iy2_B)*width + ix2_R] - image_data[(ch_off + iy2_T)*width + ix2_R]);

            bot_diff += warped_diff[(ch_off + y)*width + x] * temp;
        }
        flow_diff[((2*n+1)*height + y)*width + x] = bot_diff;
    }
}


template <typename Dtype>
__global__ void flow_warp_backward_kernel_smem(const Dtype* trans_image_data, Dtype* image_diff, const Dtype* flow_data, Dtype* flow_diff, const Dtype* warped_diff, int num, int channels, int cblocks, int width, int wblocks, int height, int widthheight)
{
//    int y = blockIdx.y;
//    int n = blockIdx.z;

//    __shared__ float x2_buf[FW_TILE_X], y2_buf[FW_TILE_X];
//    __shared__ float buffer[FW_TILE_C][FW_TILE_X+1];

//    int x;
//    int c;

//    x = blockIdx.x*FW_TILE_X + threadIdx.x;
//    if(threadIdx.y==0 && x<width)
//    {
//        x2_buf[threadIdx.x] = float(x) + flow[((2*n  )*height + y)*width + x];
//        y2_buf[threadIdx.x] = float(y) + flow[((2*n+1)*height + y)*width + x];
//    }

//    __syncthreads();

//    float x2 = x2_buf[threadIdx.y];
//    float y2 = y2_buf[threadIdx.y];

//    int ix2_L = int(x2);
//    int iy2_T = int(y2);
//    int ix2_R = min(ix2_L+1, width-1);
//    int iy2_B = min(iy2_T+1, height-1);

//    int off_TL = ((n*height + iy2_T)*width + ix2_L)*channels;
//    int off_TR = ((n*height + iy2_T)*width + ix2_R)*channels;
//    int off_BL = ((n*height + iy2_B)*width + ix2_L)*channels;
//    int off_BR = ((n*height + iy2_B)*width + ix2_R)*channels;

//    float alpha = x2-ix2_L;
//    float beta = y2-iy2_T;
//    float coeffTL = (1-alpha)*(1-beta);
//    float coeffTR = alpha*(1-beta);
//    float coeffBL = (1-alpha)*beta;
//    float coeffBR = alpha*beta;

//    for(int cb=0; cb<cblocks; cb++)
//    {
//        __syncthreads();

//        buffer[threadIdx.y][threadIdx.x] = 0;

//        __syncthreads();

//        c = cb*FW_TILE_C + threadIdx.y;
//        x = blockIdx.x*FW_TILE_X + threadIdx.x;
//        if(c<channels && x<width)
//            buffer[threadIdx.y][threadIdx.x] = warped_diff[((n*channels + c)*height + y)*width + x]; // buffer[c][x]

//        __syncthreads();

//        c = cb*FW_TILE_C + threadIdx.x;
//        float wd = buffer[threadIdx.x][threadIdx.y];
//        if(x2>=0 && y2>=0 && x2<width && y2<height && c<channels && x<width)
//        {
//            atomicAdd(&image_diff[((n*channels + c)*height + iy2_T)*width + ix2_L], wd * coeffTL);
//            atomicAdd(&image_diff[((n*channels + c)*height + iy2_T)*width + ix2_R], wd * coeffTR);
//            atomicAdd(&image_diff[((n*channels + c)*height + iy2_B)*width + ix2_L], wd * coeffBL);
//            atomicAdd(&image_diff[((n*channels + c)*height + iy2_B)*width + ix2_R], wd * coeffBR);

//            float gamma = iy2_B - y2;
//            c = cb*FW_TILE_C + threadIdx.x;
//            float imgTR = trans_image_data[((n*height + iy2_T)*width + ix2_R)*channels + c];
//            float imgTL = trans_image_data[((n*height + iy2_T)*width + ix2_L)*channels + c];
//            float imgBR = trans_image_data[((n*height + iy2_B)*width + ix2_R)*channels + c];
//            float imgBL = trans_image_data[((n*height + iy2_B)*width + ix2_L)*channels + c];

//            float temp = 0;
//            temp += gamma     * (imgTR - imgTL);
//            temp += (1-gamma) * (imgBR - imhBL);
//            temp *= buffer[threadIdx.x][threadIdx.y]; // warped_diff[((n*channels + c)*height + y)*width + x]
//            atomicAdd(&flow_diff[(2*n*height + y)*width + x], wd * coeffBR);

//        }



//        for(int c=0; c<channels; c++)
//        {
//            float temp = 0;
//            temp += gamma     * (imgTR - imgTL);
//            temp += (1-gamma) * (imgBR - imhBL);

//            bot_diff += warped_diff[((n*channels + c)*height + y)*width + x] * temp;
//        }
//        flow_diff[(2*n*height + y)*width + x] = bot_diff;

//        gamma = ix2_R - x2;
//        bot_diff = 0;
//        for(int c=0; c<channels; c++)
//        {
//            float temp = 0;
//            temp += gamma *     (image_data[((n*channels + c)*height + iy2_B)*width + ix2_L] - image_data[((n*channels + c)*height + iy2_T)*width + ix2_L]);
//            temp += (1-gamma) * (image_data[((n*channels + c)*height + iy2_B)*width + ix2_R] - image_data[((n*channels + c)*height + iy2_T)*width + ix2_R]);

//            bot_diff += warped_diff[((n*channels + c)*height + y)*width + x] * temp;
//        }
//        flow_diff[((2*n+1)*height + y)*width + x] = bot_diff;



//        int c = cb*FW_TILE_C + threadIdx.x;
//        if(x2>=0 && y2>=0 && x2<width && y2<height && c<channels)
//            buffer[threadIdx.y][threadIdx.x] =  // buffer [x][c]
//                coeffTL * image[off_TL + c] +
//                coeffTR * image[off_TR + c] +
//                coeffBL * image[off_BL + c] +
//                coeffBR * image[off_BR + c];

//        __syncthreads();

//        c = cb*FW_TILE_C + threadIdx.y;
//        int x = blockIdx.x*FW_TILE_X + threadIdx.x;
//        if(c<channels && x<width)
//            warped[((n*channels+c)*height + y)*width + x] = buffer[threadIdx.x][threadIdx.y];
//    }
}



template <typename Dtype>
void FlowWarpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    int width = top[0]->width();
    int height = top[0]->height();
    int channels = top[0]->channels();
    int num = top[0]->num();
    const int wh_size = width * height;
    const int whc_size = width * height * channels;

    Dtype* warped_data = top[0]->mutable_gpu_data(); // dest
    const Dtype* image_data = bottom[0]->gpu_data(); // source image
    Dtype* trans_image_data = transposed_image_.mutable_gpu_data(); // source image
    const Dtype* flow_data = bottom[1]->gpu_data(); // source flow

    int nan = 0xFFE00000;
    float nanf = *(reinterpret_cast<float*>(&nan));

    Dtype fillValue = this->layer_param().flow_warp_param().fill_value() == FlowWarpParameter_FillParameter_ZERO ? 0 : nanf;

    cudaMemset(warped_data, fillValue, width*height*channels*num*sizeof(float));

#ifdef DISPLAY_TIMINGS
    caffe::Timer t1;
    t1.Start();
#endif
    dim3 rearrangeThreads(RA_TILE,RA_ROWS,1);
    int cblocks = ((channels-1)/RA_TILE+1);
    dim3 rearrangeBlocks(cblocks*num, (width-1)/RA_TILE+1, height);
    flow_warp_rearrange_kernel<Dtype><<<rearrangeBlocks, rearrangeThreads>>>(
        image_data,
        trans_image_data,
        num,
        channels,
        cblocks,
        width,
        height,
        wh_size
    );
    CUDA_POST_KERNEL_CHECK;
#ifdef DISPLAY_TIMINGS
    t1.Stop();
    LOG(INFO) << "rearrange time " << t1.MilliSeconds() << "ms";
#endif

//    if(channels>8)
    {
#ifdef DISPLAY_TIMINGS
        caffe::Timer t2;
        t2.Start();
#endif
        int wblocks = ((width-1)/FW_TILE_X+1);
        int cblocks = ((channels-1)/FW_TILE_C+1);
        dim3 warpThreads(FW_TILE_X,FW_TILE_C);
        dim3 warpBlocks(wblocks, height, num);
        flow_warp_kernel_smem<Dtype><<<warpBlocks, warpThreads>>>(
            trans_image_data,
            flow_data,
            warped_data,
            num,
            channels,
            cblocks,
            width,
            wblocks,
            height,
            wh_size,
            fillValue
        );
        CUDA_POST_KERNEL_CHECK;
#ifdef DISPLAY_TIMINGS
        t2.Stop();
        LOG(INFO) << "warp time 1a: " << t2.MilliSeconds() << "ms";
#endif
    }
//    else
//    {
//#ifdef DISPLAY_TIMINGS
//        caffe::Timer t2a;
//        t2a.Start();
//#endif
//        int wblocks = ((width-1)/FW_TILE_X+1);
//        dim3 warpThreads(FW_TILE_X);
//        dim3 warpBlocks(wblocks, height, num);
//        flow_warp_kernel_no_smem<Dtype><<<warpBlocks, warpThreads>>>(
//            image_data,
//            flow_data,
//            warped_data,
//            num,
//            channels,
//            width,
//            wblocks,
//            height,
//            wh_size
//        );
//        CUDA_POST_KERNEL_CHECK;
//#ifdef DISPLAY_TIMINGS
//        t2a.Stop();
//        LOG(INFO) << "warp time 1b: " << t2a.MilliSeconds() << "ms";
//#endif
//    }

}

template <typename Dtype>
void FlowWarpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    int width = top[0]->width();
    int height = top[0]->height();
    int channels = top[0]->channels();
    int num = top[0]->num();
    const int wh_size = width * height;
    const int whc_size = width * height * channels;

    const Dtype* warped_data = top[0]->gpu_data(); // dest
    const Dtype* warped_diff = top[0]->gpu_diff(); // dest
    const Dtype* image_data = bottom[0]->gpu_data(); // source image
    Dtype* image_diff = bottom[0]->mutable_gpu_diff(); // source image
    const Dtype* flow_data = bottom[1]->gpu_data(); // source flow
    Dtype* flow_diff = bottom[1]->mutable_gpu_diff(); // source flow

    cudaMemset(image_diff, 0, width*height*channels*num*sizeof(float));
    cudaMemset(flow_diff, 0, width*height*2*num*sizeof(float));

    //Backward_cpu(top, propagate_down, bottom);
    //return;

#ifdef DISPLAY_TIMINGS
    caffe::Timer t3a;
    t3a.Start();
#endif
    int wblocks = ((width-1)/FW_TILE_X+1);
    int cblocks = ((channels-1)/FW_TILE_C+1);
    dim3 warpThreads(FW_TILE_X,1);
    dim3 warpBlocks(wblocks, height, num);
    flow_warp_backward_kernel_no_smem<Dtype><<<warpBlocks, warpThreads>>>(
        image_data,
        (float*)image_diff,
        flow_data,
        flow_diff,
        warped_diff,
        num,
        channels,
        cblocks,
        width,
        wblocks,
        height,
        wh_size
    );
    CUDA_POST_KERNEL_CHECK;
#ifdef DISPLAY_TIMINGS
    t3a.Stop();
    LOG(INFO) << "backward time 1a: " << t3a.MilliSeconds() << "ms";
#endif

    if(!propagate_down[0]) caffe_gpu_memset(bottom[0]->count()*sizeof(Dtype), 0, image_diff);
    if(!propagate_down[1]) caffe_gpu_memset(bottom[1]->count()*sizeof(Dtype), 0, flow_diff);

//    {
//        printf("gpu flow u:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[1]->data_at(0, 0, y, x));
//            }
//            printf("\n");
//        }
//        printf("gpu flow v:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[1]->data_at(0, 1, y, x));
//            }
//            printf("\n");
//        }
//        printf("gpu image:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[0]->data_at(0, 0, y, x));
//            }
//            printf("\n");
//        }
//        printf("gpu flow diff u:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[1]->diff_at(0, 0, y, x));
//            }
//            printf("\n");
//        }
//        printf("gpu flow diff v:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[1]->diff_at(0, 1, y, x));
//            }
//            printf("\n");
//        }
//        printf("gpu image diff:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[0]->diff_at(0, 0, y, x));
//            }
//            printf("\n");
//        }
//    }
}


INSTANTIATE_LAYER_GPU_FUNCS(FlowWarpLayer);



}  // namespace caffe


















//caffe::Timer t3;
//t3.Start();

//int topcount = width*height*channels*num;
//WarpData<Dtype><<<CAFFE_GET_BLOCKS(topcount), CAFFE_CUDA_NUM_THREADS>>>(
//    topcount,
//    num, channels, height, width, image_data, topcount,
//    height, width, warped_data, flow_data);

//t3.Stop();
//LOG(INFO) << "warp time 2: " << t3.MilliSeconds() << "ms";



//template <typename Dtype>
//__global__ void WarpData(const int nthreads, const int num, const int channels, const int height, const int width, const Dtype* src_data, const int src_count,
//                                const int dest_height, const int dest_width, Dtype* dest_data, const Dtype* flow) {
//  CUDA_KERNEL_LOOP(index, nthreads) {
//    int x = (index % width); //w-pos
//    int y = ((index / width) % height); //h-pos
//    int cn = (index / width / height); // channel*num
//    int n = cn / channels; //num

//    // === Warping:

//    float xpos = (float)(x) + flow[width*(height*(2*n+0) + y) + x];
//    float ypos = (float)(y) + flow[width*(height*(2*n+1) + y) + x];

//    if (xpos > 0.f && xpos <= width-1.01f && ypos > 0.f && ypos <= height-1.01f) {
//      // Get interpolated sample
//      float tlx = floor(xpos);
//      float tly = floor(ypos);
//      int srcIdxOff = width*(height*cn + tly) + tlx;

//      float sampleTL = src_data[srcIdxOff];
//      float sampleTR = src_data[min(srcIdxOff+1,src_count)];
//      float sampleBL = src_data[min(srcIdxOff+width,src_count)];
//      float sampleBR = src_data[min(srcIdxOff+1+width,src_count)];

//      float xdist = xpos - tlx;
//      float ydist = ypos - tly;

//      float sample = (1-xdist)*(1-ydist)*sampleTL
//                  + (  xdist)*(  ydist)*sampleBR
//                  + (1-xdist)*(  ydist)*sampleBL
//                  + (  xdist)*(1-ydist)*sampleTR;

//      dest_data[index] = sample;
//    }
//  }
//}




//volatile float TL = image[((n*channels + c)*height + iy2_T)*width + ix2_L];
//volatile float TR = image[((n*channels + c)*height + iy2_T)*width + ix2_R];
//volatile float BL = image[((n*channels + c)*height + iy2_B)*width + ix2_L];
//volatile float BR = image[((n*channels + c)*height + iy2_B)*width + ix2_R];

//threadIdx.y;

//    if(threadIdx.y == 0)
//    {
//        x2[xoff] = float(x) + flow[((2*n  )*height + y)*width + x];
//        y2[xoff] = float(y) + flow[((2*n+1)*height + y)*width + x];
//    }

//    __syncthreads();



//        __syncthreads();

//        if(x2>=0 && y2>=0 && x2<width && y2<height)
//        {
//            int ix2_L = int(x2);
//            int iy2_T = int(y2);
//            int ix2_R = min(ix2_L+1, width-1);
//            int iy2_B = min(iy2_T+1, height-1);

//            float alpha=x2-ix2_L;
//            float beta=y2-iy2_T;

//            for(int c=threadIdx.x; c<channels; c+=blockDim.x)
//            {
//                float TL = image[((n*height + iy2_T)*width + ix2_L)*channels + c];
//                float TR = image[((n*height + iy2_T)*width + ix2_R)*channels + c];
//                float BL = image[((n*height + iy2_B)*width + ix2_L)*channels + c];
//                float BR = image[((n*height + iy2_B)*width + ix2_R)*channels + c];

//                //warped[((n*height+y)*width + x)*channels + c] =
//                warped[((n*channels+c)*height + y)*width + x] =
//                    (1-alpha)*(1-beta)*TL +
//                    alpha*(1-beta)*TR +
//                    (1-alpha)*beta*BL +
//                    alpha*beta*BR;
//            }
//        }
//        else
//        {
//            for(int c=threadIdx.x; c<channels; c+=blockDim.x)
//                warped[((n*channels+c)*height + y)*width + x] = 0;
//        }
//    }

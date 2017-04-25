// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/channel_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <fstream>

#define CUDART_NAN_F            __int_as_float(0x7fffffff)

namespace caffe {

template <typename Dtype> 
__global__ void NormForward(const int count,const int num, const int channels, const int height, const int width, const Dtype* bot_data, Dtype* top_data)
{
  CUDA_KERNEL_LOOP(index, count) {
    int x = (index % width); //w-pos
    int y = (index / width) % height; //h-pos
    int n = (index / width / height); //num

    Dtype norm = 0;
    for(int c=0; c<channels; c++)
    {
        Dtype val = bot_data[((n*channels+c)*height+y)*width+x];
        norm += val*val;
    }

    norm = sqrt(norm);
    top_data[index] = norm;
  }
}

template <typename Dtype>
__global__ void NormBackward(const int count,const int num, const int channels, const int height, const int width, const Dtype* bot_data, Dtype* bot_diff, const Dtype* top_data, const Dtype* top_diff)
{
  CUDA_KERNEL_LOOP(index, count) {
    int x = (index % width);
    int y = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = (index / width / height / channels);

    bot_diff[((n*channels+c)*height+y)*width+x] = top_diff[(n*height+y)*width+x] * bot_data[((n*channels+c)*height+y)*width+x] / (top_data[(n*height+y)*width+x]+1e-9);
  }
}


template <typename Dtype>
void ChannelNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    int width = top[0]->width();
    int height = top[0]->height();
    int channels = bottom[0]->channels();
    int num = top[0]->num();

    Dtype* top_data = top[0]->mutable_gpu_data(); // dest
    const Dtype* bot_data = bottom[0]->gpu_data(); // source image

    NormForward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(),
        num, channels, height, width, bot_data,
        top_data
    );

    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ChannelNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    int width = top[0]->width();
    int height = top[0]->height();
    int channels = bottom[0]->channels();
    int num = top[0]->num();

    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bot_data = bottom[0]->cpu_data();
    Dtype* bot_diff = bottom[0]->mutable_cpu_diff();

    NormBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(),
        num, channels, height, width,
        bot_data, bot_diff, top_data, top_diff
    );
}


INSTANTIATE_LAYER_GPU_FUNCS(ChannelNormLayer);

}  // namespace caffe

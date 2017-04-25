// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/channel_norm_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {

template <typename Dtype>
void ChannelNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ChannelNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "ChannelNormLayer takes two input blobs: image and flow.";
  CHECK_EQ(top.size(), 1) << "ChannelNormLayer outputs one blob.";
  
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
   // = Allocate output
  top[0]->Reshape(num, 1, height, width);
}

template <typename Dtype>
void ChannelNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    int width = top[0]->width();
    int height = top[0]->height();
    int channels = bottom[0]->channels();
    int num = top[0]->num();

    Dtype* top_data = top[0]->mutable_cpu_data(); // dest
    const Dtype* bot_data = bottom[0]->cpu_data(); // source image

    memset(top_data, 0, width*height*num*sizeof(Dtype));
    for(int n=0; n<num; n++)
        for(int x=0; x<width; x++)
            for(int y=0; y<height; y++)
            {
                Dtype norm = 0;

                for(int c=0; c<channels; c++)
                {
                    Dtype val = bot_data[((n*channels+c)*height+y)*width+x];
                    norm += val*val;
                }

                norm = sqrt(norm);
                top_data[(n*height+y)*width+x] = norm;
            }

//    printf("forward bot_data:\n");
//    for(int n=0; n<num; n++)
//    {
//        for(int c=0; c<channels; c++)
//            for(int y=0; y<height; y++)
//            {
//                for(int x=0; x<width; x++)
//                {
//                    printf("%f ", bottom[0]->data_at(n, c, y, x));
//                }
//                printf("\n");
//            }
//        printf("\n");
//    }

//    printf("forward top_data:\n");
//    for(int n=0; n<num; n++)
//    {
//        for(int c=0; c<1; c++)
//            for(int y=0; y<height; y++)
//            {
//                for(int x=0; x<width; x++)
//                {
//                    printf("%f ", top[0]->data_at(n, c, y, x));
//                }
//                printf("\n");
//            }
//        printf("\n");
//    }
}

template <typename Dtype>
void ChannelNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    //if(!propagate_down[0])
        //return;

    int width = top[0]->width();
    int height = top[0]->height();
    int channels = bottom[0]->channels();
    int num = top[0]->num();

    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bot_data = bottom[0]->cpu_data();
    Dtype* bot_diff = bottom[0]->mutable_cpu_diff();

    memset(bot_diff, 0, width*height*channels*num*sizeof(Dtype));
    for(int n=0; n<num; n++)
        for(int x=0; x<width; x++)
            for(int y=0; y<height; y++)
                for(int c=0; c<channels; c++)
                    bot_diff[((n*channels+c)*height+y)*width+x] = top_diff[(n*height+y)*width+x] * bot_data[((n*channels+c)*height+y)*width+x] / (top_data[(n*height+y)*width+x]+1e-9);

//    printf("backward top_data:\n");
//    for(int n=0; n<num; n++)
//    {
//        for(int c=0; c<1; c++)
//            for(int y=0; y<height; y++)
//            {
//                for(int x=0; x<width; x++)
//                {
//                    printf("%f ", top[0]->data_at(n, c, y, x));
//                }
//                printf("\n");
//            }
//        printf("\n");
//    }

//    printf("backward top_diff:\n");
//    for(int n=0; n<num; n++)
//    {
//        for(int c=0; c<1; c++)
//            for(int y=0; y<height; y++)
//            {
//                for(int x=0; x<width; x++)
//                {
//                    printf("%f ", top[0]->diff_at(n, c, y, x));
//                }
//                printf("\n");
//            }
//        printf("\n");
//    }

//    printf("backward bot_data:\n");
//    for(int n=0; n<num; n++)
//    {
//        for(int c=0; c<channels; c++)
//            for(int y=0; y<height; y++)
//            {
//                for(int x=0; x<width; x++)
//                {
//                    printf("%f ", bottom[0]->data_at(n, c, y, x));
//                }
//                printf("\n");
//            }
//        printf("\n");
//    }

//    printf("backward bot_diff:\n");
//    for(int n=0; n<num; n++)
//    {
//        for(int c=0; c<channels; c++)
//            for(int y=0; y<height; y++)
//            {
//                for(int x=0; x<width; x++)
//                {
//                    printf("%f ", bottom[0]->diff_at(n, c, y, x));
//                }
//                printf("\n");
//            }
//        printf("\n");
//    }

}


#ifdef CPU_ONLY
STUB_GPU(ChannelNormLayer);
#endif

INSTANTIATE_CLASS(ChannelNormLayer);
REGISTER_LAYER_CLASS(ChannelNorm);


}  // namespace caffe

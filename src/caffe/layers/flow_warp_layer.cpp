// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>
#include <stdio.h>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/flow_warp_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {

template <typename Dtype>
void FlowWarpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void FlowWarpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2) << "FlowWarpLayer takes two input blobs: image and flow.";
  CHECK_EQ(top.size(), 1) << "FlowWarpLayer outputs one blob.";
  
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
  CHECK_EQ(num, bottom[1]->num()) << "Num of the inputs should be the same";
  CHECK_EQ(2, bottom[1]->channels()) << "Flow should have 2 channels: x-flow and y-flow";
  CHECK_EQ(width, bottom[1]->width()) << "Width of the inputs should be the same";
  CHECK_EQ(height, bottom[1]->height()) << "Height of the inputs should be the same";  
  
  top[0]->Reshape(num, channels, height, width);  
  transposed_image_.Reshape(num, height, width, channels);
}

#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

template <typename Dtype>
void FlowWarpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    int width = top[0]->width();
    int height = top[0]->height();
    int channels = top[0]->channels();
    int num = top[0]->num();
    const int wh_size = width * height;
    const int whc_size = width * height * channels;

    Dtype* warped_data = top[0]->mutable_cpu_data(); // dest
    const Dtype* image_data = bottom[0]->cpu_data(); // source image
    const Dtype* flow_data = bottom[1]->cpu_data(); // source flow

    Dtype fillValue = this->layer_param().flow_warp_param().fill_value() == FlowWarpParameter_FillParameter_ZERO ? 0 : NAN;

    for(int n=0; n<num; n++)
    {
        int off = whc_size * n;
        for(int x=0; x<width; x++)
            for(int y=0; y<height; y++)
            {
                float fx = flow_data[2*wh_size*n + y*width + x];
                float fy = flow_data[2*wh_size*n + wh_size + y*width + x];

                float x2 = float(x) + fx;
                float y2 = float(y) + fy;

                if(x2>=0 && y2>=0 && x2<width && y2<height)
                {
                    int ix2_L = int(x2);
                    int iy2_T = int(y2);
                    int ix2_R = min(ix2_L+1, width-1);
                    int iy2_B = min(iy2_T+1, height-1);

                    float alpha=x2-ix2_L;
                    float beta=y2-iy2_T;

                    for(int c=0; c<channels; c++)
                    {
                        float TL = image_data[off + c*wh_size + iy2_T*width + ix2_L];
                        float TR = image_data[off + c*wh_size + iy2_T*width + ix2_R];
                        float BL = image_data[off + c*wh_size + iy2_B*width + ix2_L];
                        float BR = image_data[off + c*wh_size + iy2_B*width + ix2_R];

                        warped_data[off + c*wh_size + y*width + x] =
                            (1-alpha)*(1-beta)*TL +
                            alpha*(1-beta)*TR +
                            (1-alpha)*beta*BL +
                            alpha*beta*BR;
                    }
                }
                else
                {
                    for(int c=0; c<channels; c++)
                        warped_data[off + c*wh_size + y*width + x] = fillValue;
                }
            }
    }
}

template <typename Dtype>
void FlowWarpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    int width = top[0]->width();
    int height = top[0]->height();
    int channels = top[0]->channels();
    int num = top[0]->num();
    const int wh_size = width * height;
    const int whc_size = width * height * channels;

    const Dtype* warped_data = top[0]->cpu_data(); // dest
    const Dtype* warped_diff = top[0]->cpu_diff(); // dest
    const Dtype* image_data = bottom[0]->cpu_data(); // source image
    Dtype* image_diff = bottom[0]->mutable_cpu_diff(); // source image
    const Dtype* flow_data = bottom[1]->cpu_data(); // source flow
    Dtype* flow_diff = bottom[1]->mutable_cpu_diff(); // source flow

    for(int i=0; i<num*whc_size; i++)
        image_diff[i] = 0 ;

    for(int n=0; n<num; n++)
    {
        int off = whc_size * n;
        for(int x=0; x<width; x++)
            for(int y=0; y<height; y++)
            {
                float fx = flow_data[2*wh_size*n + y*width + x];
                float fy = flow_data[2*wh_size*n + wh_size + y*width + x];

                float x2 = float(x) + fx;
                float y2 = float(y) + fy;

                if(x2>=0 && y2>=0 && x2<width && y2<height)
                {
                    int ix2_L = int(x2);
                    int iy2_T = int(y2);
                    int ix2_R = min(ix2_L+1, width-1);
                    int iy2_B = min(iy2_T+1, height-1);

                    float alpha=x2-ix2_L;
                    float beta=y2-iy2_T;
                    for(int c=0; c<channels; c++)
                    {
                        float warped_diff_value = warped_diff[off + c*wh_size + y*width + x];
                        image_diff[off + c*wh_size + iy2_T*width + ix2_L] += warped_diff_value * (1-alpha)*(1-beta);
                        image_diff[off + c*wh_size + iy2_T*width + ix2_R] += warped_diff_value * alpha*(1-beta);
                        image_diff[off + c*wh_size + iy2_B*width + ix2_L] += warped_diff_value * (1-alpha)*beta;
                        image_diff[off + c*wh_size + iy2_B*width + ix2_R] += warped_diff_value * alpha*beta;
                    }

                    float gamma = iy2_B - y2;
                    float bot_diff = 0;
                    for(int c=0; c<channels; c++)
                    {
                        float temp = 0;
                        temp += gamma *     (image_data[off + c*wh_size + iy2_T*width + ix2_R] - image_data[off + c*wh_size + iy2_T*width + ix2_L]);
                        temp += (1-gamma) * (image_data[off + c*wh_size + iy2_B*width + ix2_R] - image_data[off + c*wh_size + iy2_B*width + ix2_L]);

                        bot_diff += warped_diff[off + c*wh_size + y*width + x] * temp;
                    }
                    flow_diff[2*wh_size*n + y*width + x] = bot_diff;

                    gamma = ix2_R - x2;
                    bot_diff = 0;
                    for(int c=0; c<channels; c++)
                    {
                        float temp = 0;
                        temp += gamma *     (image_data[off + c*wh_size + iy2_B*width + ix2_L] - image_data[off + c*wh_size + iy2_T*width + ix2_L]);
                        temp += (1-gamma) * (image_data[off + c*wh_size + iy2_B*width + ix2_R] - image_data[off + c*wh_size + iy2_T*width + ix2_R]);

                        bot_diff += warped_diff[off + c*wh_size + y*width + x] * temp;
                    }
                    flow_diff[2*wh_size*n + wh_size + y*width + x] = bot_diff;
                }
            }
    }

    if(!propagate_down[0]) caffe_memset(bottom[0]->count()*sizeof(Dtype), 0, image_diff);
    if(!propagate_down[1]) caffe_memset(bottom[1]->count()*sizeof(Dtype), 0, flow_diff);


//    {
//        printf("cpu flow u:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[1]->data_at(0, 0, y, x));
//            }
//            printf("\n");
//        }
//        printf("cpu flow v:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[1]->data_at(0, 1, y, x));
//            }
//            printf("\n");
//        }
//        printf("cpu image:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[0]->data_at(0, 0, y, x));
//            }
//            printf("\n");
//        }
//        printf("cpu flow diff u:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[1]->diff_at(0, 0, y, x));
//            }
//            printf("\n");
//        }
//        printf("cpu flow diff v:\n");
//        for(int y=0; y<height; y++)
//        {
//            for(int x=0; x<width; x++)
//            {
//                printf("%f ", bottom[1]->diff_at(0, 1, y, x));
//            }
//            printf("\n");
//        }
//        printf("cpu image diff:\n");
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

INSTANTIATE_CLASS(FlowWarpLayer);
REGISTER_LAYER_CLASS(FlowWarp);


}  // namespace caffe


























//                    }
//                }
//            }

//        for(int x=0; x<width; x++)
//            for(int y=0; y<height; y++)
//            {
//                float fx = flow_data[off + y*width + x];
//                float fy = flow_data[off + wh_size + y*width + x];

//                float x2 = float(x) + fx;
//                float y2 = float(y) + fy;

//                if(x2>=0 && y2>=0 && x2<width && y2<height)
//                {
//                    int ix2_L = int(x2);
//                    int iy2_T = int(y2);
//                    int ix2_R = min(ix2_L+1, width-1);
//                    int iy2_B = min(iy2_T+1, height-1);

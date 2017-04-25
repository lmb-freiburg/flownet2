// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/downsample_layer.hpp"
#include "caffe/util/math_functions.hpp"


#define CUDART_NAN_F            __int_as_float(0x7fffffff)

namespace caffe {

template <typename Dtype>
__global__ void DownsampleFeatures(const int nthreads, const int num, const int channels, const int bottomwidth, const int bottomheight,
            const int topheight, const int topwidth, const int bot_countpernum, const int bot_numstride, const float widthScale, const float heightScale, const int wradius, const int hradius, const Dtype* src_data, Dtype* dest_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // From top (large,src_data) to bottom (small,dst_data)
    
    int destx = index % topwidth; //w-pos
    int desty = (index / topwidth) % topheight; //h-pos
    
    int cn = (index / topwidth / topheight);
    int c = cn % channels; //channel
    int n = cn / channels; //num
    
    //Compute source center pos in topdiff
    float botx = ((float)destx/(float)(topwidth-1)) * (float)(bottomwidth-1); // \in [0.0, (topwidth-1)]
    float boty = ((float)desty/(float)(topheight-1)) * (float)(bottomheight-1);
    
    int ibotx = round(botx);
    int iboty = round(boty);
    
    //printf("dest x,y[%d,%d] n:%d c:%d = idx:%d/%d | bot x,y[%d,%d]\n", destx, desty, n, c, index, nthreads, ibotx, iboty);
    
    // Accumulate in range around that point:
    int botidxoffcn = (bot_numstride*n) + (bottomwidth*bottomheight*c);
    
    float accum_value = 0;
    float accum_weight = 0;
    float accum_nan = 0;
    
    for(int yoff = -hradius; yoff <= hradius; yoff++)  {
        int by = iboty + yoff;
        int botidxoffycn = by*bottomwidth + botidxoffcn;
        for(int xoff = -wradius; xoff <= wradius; xoff++)  {
            int bx = ibotx + xoff;
            
            if(bx >= 0 && by >= 0 && bx < bottomwidth && by < bottomheight) {
                float sample = src_data[bx + botidxoffycn];
                float weight = max(0.0f,1.0f-(abs((float)bx - botx)/widthScale)) * max(0.0f,1.0f- (abs((float)by - boty)/heightScale) );
                if(sample != sample) { //isnan
                  accum_nan += weight;
                  sample = 0;
                  weight = 0;
                }
                
                accum_value += sample * weight;
                accum_weight += weight;
            }
        }
    }
    if(accum_nan / accum_weight > 0.5) {
      dest_data[index] = CUDART_NAN_F;
    } else {
      dest_data[index] = accum_value / accum_weight;
    }
    
    //printf("dest x,y[%d,%d] n:%d c:%d = idx:%d | bot x,y[%d,%d] (int)val %d\n", destx, desty, n, c, index, ibotx, iboty, (int)src_data[ibotx + (iboty*bottomwidth + botidxoffcn)]);    
    //dest_data[index] = src_data[ibotx + (iboty*bottomwidth + botidxoffcn)];
  }
}

template <typename Dtype>
void DownsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  Dtype* top_data = top[0]->mutable_gpu_data(); // dest
  int topwidth = top[0]->width();
  int topheight = top[0]->height();
  int topchannels = top[0]->channels();
  int topcount = top[0]->count();
  
  //LOG(INFO) << "Metrics: Tnum " << top[0]->num() << " Tchan " << topchannels << " Tw " << topwidth << " Th " << topheight;
    
  const Dtype* bottom_data = bottom[0]->gpu_data(); // source
  //LOG(INFO) << "Got ptr to bottom ";
  
  int bottomnum = (bottom)[0]->num();
  int bottomchannels = (bottom)[0]->channels();
  int bottomwidth = (bottom)[0]->width();
  int bottomheight = (bottom)[0]->height();
  int bottomcount = (bottom)[0]->count();
  
  if (bottomwidth != topwidth || bottomheight != topheight) { 
    
    // From bottom to top
    
    int bot_countpernum = bottomwidth * bottomheight * bottomchannels;
    int bot_numstride = bottomwidth * bottomheight * bottomchannels;
    
    float widthScale = (float)(bottomwidth-1) / (float)(topwidth-1); // e.g. 2.0 if bottom pixeldist half compared to top. 
    float heightScale = (float)(bottomheight-1) / (float)(topheight-1);
    
    const int wradius = ceil(widthScale); //One pixel from bottom is incfluenced by +- widthScale or heightScale pixels around that in top 
    const int hradius = ceil(heightScale);
    
    // Loop over: bottomwidth,bottomheight,bottomlayers. (x,y,l)
    // Accumulate data from top_diff_chanoffsetptr, at
    //  topx = (x/(bottomwidth-1)) [0.0, 1.0]  * (topwidth-1) = [0.0, (topwidth-1)]
    //  topy = analogously
    // in a rectangle around that point width range [-wradius,+wradius][-hradius,+hradius]
    // and weight each toppixel with "closeness" (=max(0,1-xdist)*max(0,1-ydist)) to [topx,topy] but xdist and ydist scaled by widthScale and heightScale.
    
    //LOG(INFO) << "Metrics: Bnum " << bottomnum << " Bchan " << bottomchannels << " Bw " << bottomwidth << " Bh " << bottomheight << " widSc " << widthScale << " hSc " << heightScale << " wrad " << wradius << " hrad " << hradius;
  
  
//     caffe_gpu_memcpy(topcount * sizeof(Dtype), bottom_data, top_data);      
    DownsampleFeatures<Dtype><<<CAFFE_GET_BLOCKS(topcount), CAFFE_CUDA_NUM_THREADS>>>(
            topcount,
            bottomnum, bottomchannels, bottomwidth, bottomheight, 
            topheight, topwidth, bot_countpernum, bot_numstride, widthScale, heightScale, wradius, hradius, bottom_data, top_data);
    
    CUDA_POST_KERNEL_CHECK;
  }
  
  //*top_data = *bottom_data;
}


template <typename Dtype>
void DownsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for(int i=0; i<propagate_down.size(); i++) 
	if(propagate_down[i]) 
		LOG(FATAL) << "DownsamplingLayer cannot do backward.";
}

INSTANTIATE_LAYER_GPU_FUNCS(DownsampleLayer);

}  // namespace caffe

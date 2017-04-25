// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/accum_layer.hpp"

namespace caffe {

template <typename Dtype> 
__global__ void UpscaleFeatures(const int nthreads, const int num, const int channels, const int height, const int width, const Dtype* src_data, const int src_count,
                                const int dest_height, const int dest_width, const int dest_countpernum, const int dest_numstride, Dtype* dest_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int dw = index % dest_width; //w-pos
    int dh = (index / dest_width) % dest_height; //h-pos
    
    int cn = (index / dest_width / dest_height);
    //int c = cn % channels; //channel
    int n = cn / channels; //num
    
    float xpos = dw / (float)(dest_width-1) * (float)(width-1);
    float ypos = dh / (float)(dest_height-1) * (float)(height-1);
    
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
    
    int destindex = index % dest_countpernum + (dest_numstride * n);
    dest_data[destindex] = sample;
  }
}

template <typename Dtype>
void AccumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Dtype* top_data = top[0]->mutable_gpu_data(); // dest
  int topwidth = top[0]->width();
  int topheight = top[0]->height();
  int topchannels = top[0]->channels();
  
  int nbottom=bottom.size();
  if(this->layer_param_.accum_param().have_reference())
      nbottom--;

  int cntChannelsOfPrevious = 0;
  for (int i = 0; i < nbottom; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data(); // source
    int bottomcount = bottom[i]->count();
    
    int bottomnum = bottom[i]->num();
    int bottomchannels = bottom[i]->channels();
    int bottomwidth = bottom[i]->width();
    int bottomheight = bottom[i]->height();
    
    //int bottomlayers = bottomnum*bottomchannels;
    
    //myparms.srcPtr = make_cudaPitchedPtr((void *)bottom_data, bottomwidth * sizeof(float), bottomwidth, bottomheight);
    
    Dtype* top_data_chanoffsetptr = top_data + (top[0]->offset(0, cntChannelsOfPrevious));
    int topcount = topwidth * topheight * bottomchannels * bottomnum;
    
    // Number of elements in toplayer sub-part corresponding to current bottom[i]
    int top_countpernum = topwidth * topheight * bottomchannels;
    // Number of elements in one toplayer disregarding num. This is the amount of entries to skip from one num to the next
    int top_numstride = topwidth * topheight * topchannels;
    
    UpscaleFeatures<Dtype><<<CAFFE_GET_BLOCKS(topcount), CAFFE_CUDA_NUM_THREADS>>>(
            topcount, bottomnum, bottomchannels, bottomheight, bottomwidth, bottom_data, bottomcount,
            topheight, topwidth, top_countpernum, top_numstride, top_data_chanoffsetptr);
    
    CUDA_POST_KERNEL_CHECK;
    
    cntChannelsOfPrevious += bottom[i]->channels();
  }
  
  CHECK_EQ(cntChannelsOfPrevious, top[0]->channels()); //Should have processed all top channels
}


template <typename Dtype>
__global__ void DownscaleFeatures(const int nthreads, const int bottomnum, const int bottomchannels, const int bottomwidth, const int bottomheight,
            const int topheight, const int topwidth, const int top_countpernum, const int top_numstride, const float widthScale, const float heightScale, const int wradius, const int hradius, const Dtype* src_data, Dtype* dest_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // From top (large,src_data) to bottom (small,dst_data)
    
    int destx = index % bottomwidth; //w-pos
    int desty = (index / bottomwidth) % bottomheight; //h-pos
    
    int cn = (index / bottomwidth / bottomheight);
    int c = cn % bottomchannels; //channel
    int n = cn / bottomchannels; //num
    
    //Compute source center pos in topdiff
    float topx = ((float)destx/(float)(bottomwidth-1)) * (float)(topwidth-1); // \in [0.0, (topwidth-1)]
    float topy = ((float)desty/(float)(bottomheight-1)) * (float)(topheight-1);
    
    int itopx = round(topx);
    int itopy = round(topy);
    
    // Accumulate in range around that point:
    int topidxoffcn = (top_numstride*n) + (topwidth*topheight*c);
    
    //printf("dest x,y[%d,%d] n:%d c:%d = idx:%d | top x,y[%d,%d]\n", destx, desty, n, c, index, itopx, itopy);
    
    float accum_value = 0;
    
    for(int yoff = -hradius; yoff <= hradius; yoff++)  {
        int ty = itopy + yoff;
        int topidxoffycn = ty*topwidth + topidxoffcn;
        for(int xoff = -wradius; xoff <= wradius; xoff++)  {
            int tx = itopx + xoff;
            
            if(tx >= 0 && ty >= 0 && tx < topwidth && ty < topheight) {
                float sample = src_data[tx + topidxoffycn];
                float weight = max(0.0f,1.0f-(abs((float)tx - topx)/widthScale)) * max(0.0f,1.0f- (abs((float)ty - topy)/heightScale) );
                
                accum_value += sample * weight;
                /*if(n == 0 && c == 0 && destx == 1 && desty == 2) {
                    printf("[%d,%d]: weight:%f sample:%f | tx-topx:%f ty-topy:%f ws:%f hs:%f\n", tx,ty, weight, sample, (float)tx - topx, (float)ty - topy, widthScale, heightScale);
                }*/
            }
        }
    }
    dest_data[index] = accum_value;
  }
}

template <typename Dtype>
void AccumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff_cpu = top[0]->cpu_diff();
  //LOG(INFO) << "pre: top_diff_chanoffsetptr[0]" << top_diff_chanoffsetptr[0];
  //LOG(INFO) << "begin: top_diff[0] = " << top_diff_cpu[0];
    
  const Dtype* top_diff = top[0]->gpu_diff();
  
  int topwidth = top[0]->width();
  int topheight = top[0]->height();
  int topchannels = top[0]->channels();
  
  // LOG(INFO) << "Metrics: Tnum " << top[0]->num() << " Tchan " << topchannels << " Tw " << topwidth << " Th " << topheight;
    
  int nbottom=bottom.size();
  if(this->layer_param_.accum_param().have_reference())
      nbottom--;

  int cntChannelsOfPrevious = 0;
  for (int i = 0; i < nbottom; ++i) {
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    
    //LOG(INFO) << "Got mutable ptr to bottom diff " << i;
    
    int bottomnum = bottom[i]->num();
    int bottomchannels = bottom[i]->channels();
    int bottomwidth = bottom[i]->width();
    int bottomheight = bottom[i]->height();
    int bottomcount = bottom[i]->count();
    
    const Dtype* top_diff_chanoffsetptr = top_diff + (top[0]->offset(0, cntChannelsOfPrevious));
    int top_countpernum = topwidth * topheight * bottomchannels;
    int top_numstride = topwidth * topheight * topchannels;
    
    float widthScale = (float)(topwidth-1) / (float)(bottomwidth-1); // e.g. 2.0 if bottom pixeldist half compared to top. 
    float heightScale = (float)(topheight-1) / (float)(bottomheight-1);
    
    const int wradius = ceil(widthScale); //One pixel from bottom is incfluenced by +- widthScale or heightScale pixels around that in top 
    const int hradius = ceil(heightScale);
    
    // Compute bottom_diff by downsampling top_diff (top_diff_chanoffsetptr):
      
    // Loop over: bottomwidth,bottomheight,bottomlayers. (x,y,l)
    // Accumulate data from top_diff_chanoffsetptr, at
    //  topx = (x/(bottomwidth-1)) [0.0, 1.0]  * (topwidth-1) = [0.0, (topwidth-1)]
    //  topy = analogously
    // in a rectangle around that point width range [-wradius,+wradius][-hradius,+hradius]
    // and weight each toppixel with "closeness" (=max(0,1-xdist)*max(0,1-ydist)) to [topx,topy] but xdist and ydist scaled by widthScale and heightScale.
    
    //LOG(INFO) << "Metrics: Bnum " << bottomnum << " Bchan " << bottomchannels << " Bw " << bottomwidth << " Bh " << bottomheight << " widSc " << widthScale << " hSc " << heightScale << " wrad " << wradius << " hrad " << hradius;
    //LOG(INFO) << "topdiff ptr offset " << top[0]->offset(0, cntChannelsOfPrevious);
    
    //LOG(INFO) << "pre: top_diff_chanoffsetptr[0]" << top_diff_chanoffsetptr[0];
    //LOG(INFO) << "pre: top_diff[0] = " << top_diff_cpu[0];
    
    DownscaleFeatures<Dtype><<<CAFFE_GET_BLOCKS(bottomcount), CAFFE_CUDA_NUM_THREADS>>>(
            bottomcount,
            bottomnum, bottomchannels, bottomwidth, bottomheight, 
            topheight, topwidth, top_countpernum, top_numstride, widthScale, heightScale, wradius, hradius, top_diff_chanoffsetptr, bottom_diff);
    
    CUDA_POST_KERNEL_CHECK;
    
    const Dtype* bottom_diff_cpu = bottom[i]->cpu_diff();
    //LOG(INFO) << "post: bottom_diff[0]" << bottom_diff_cpu[0];
    
    cntChannelsOfPrevious += bottom[i]->channels();
  }
  
    //Relu: 
    //for (int i = 0; i < count; ++i) {
    //  bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
    //}
}

INSTANTIATE_LAYER_GPU_FUNCS(AccumLayer);

}  // namespace caffe

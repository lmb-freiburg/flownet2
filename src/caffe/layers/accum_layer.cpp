// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/accum_layer.hpp"

#define min(a,b) ((a<b)?a:b)

namespace caffe {

template <typename Dtype>
void AccumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void AccumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  num_ = bottom[0]->num();
  
  
  if (this->layer_param_.accum_param().have_reference())
  {
      //LOG(INFO) << "Using reference blob of size " << bottom[bottom.size()-1]->width() << "x" << bottom[bottom.size()-1]->height();
      CHECK_GE(bottom.size(), 2) << "Need at least two bottom blobs (one as reference)";

      int totalchannels = 0;
      for (int i = 0; i < bottom.size()-1; ++i) {
        totalchannels += bottom[i]->channels();
        CHECK_EQ(num_, bottom[i]->num()) << "All accumulated layers must have same number of images (num_)";
      }
      CHECK_GE(totalchannels, 1) << "Accumulated layers must have some channels in total";

      top_height_ = bottom[bottom.size()-1]->height();
      top_width_ = bottom[bottom.size()-1]->width();
      height_ = top_height_;
      width_ = top_width_;
      channels_ = totalchannels;
  }
  else
  {
      int maxwidth = -1;
      int maxheight = -1;
      int totalchannels = 0;

      // Find largest blob size and count total channels
      for (int i = 0; i < bottom.size(); ++i) {
        totalchannels += bottom[i]->channels();
        if (bottom[i]->height() > maxheight) maxheight = bottom[i]->height();
        if (bottom[i]->width() > maxwidth) maxwidth = bottom[i]->width();
        CHECK_EQ(num_, bottom[i]->num()) << "All accumulated layers must have same number of images (num_)";
      }
      CHECK_GE(totalchannels, 1) << "Accumulated layers must have some channels in total";

      if (this->layer_param_.accum_param().has_size_divisible_by()) {
        float sdb = static_cast<float>(this->layer_param_.accum_param().size_divisible_by());
        top_height_ = static_cast<int> (ceil(maxheight / sdb) * sdb);
        top_width_ = static_cast<int> (ceil(maxwidth / sdb) * sdb);
      } else {
        top_height_ = this->layer_param_.accum_param().top_height();
        top_width_ = this->layer_param_.accum_param().top_width();
      }
      if(top_height_ > maxheight && top_width_ > maxwidth) { // Layer can specify custom top size which is larger than default
        height_ = top_height_;
        width_ = top_width_;
      } else { // Otherwise maximum of bottom sizes will be used
        height_ = maxheight;
        width_ = maxwidth;
      }
      channels_ = totalchannels;
  }
  
  count_ = width_ * height_ * channels_ * num_;
  
  top[0]->Reshape(num_, channels_, height_, width_);
  CHECK_EQ(count_, top[0]->count());
        
}

template <typename Dtype>
void AccumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  Dtype* top_data = top[0]->mutable_cpu_data(); // dest
  int topwidth = top[0]->width();
  int topheight = top[0]->height();
  int topchannels = top[0]->channels();
  
  int nbottom=bottom.size();
  if(this->layer_param_.accum_param().have_reference())
      nbottom--;

  int cntChannelsOfPrevious = 0;
  for (int i = 0; i < nbottom; ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data(); // source
    int bottomcount = bottom[i]->count();
    
    int bottomnum = bottom[i]->num();
    int bottomchannels = bottom[i]->channels();
    int bottomwidth = bottom[i]->width();
    int bottomheight = bottom[i]->height();
    
    Dtype* top_data_chanoffsetptr = top_data + (top[0]->offset(0, cntChannelsOfPrevious));
    //int topcount = topwidth * topheight * bottomchannels * bottomnum;
    
    // Number of elements in toplayer sub-part corresponding to current bottom[i]
    //int top_countpernum = topwidth * topheight * bottomchannels;
    // Number of elements in one toplayer disregarding num. This is the amount of entries to skip from one num to the next
    int top_numstride = topwidth * topheight * topchannels;
    
    for(int n = 0; n < bottomnum; n++) {
    for(int c = 0; c < bottomchannels; c++) {
        int cn = n*bottomchannels + c;
        for(int h = 0; h < topheight; h++) {
            for(int w = 0; w < topwidth; w++) {
                            
                float xpos = w / (float)(topwidth-1) * (float)(bottomwidth-1);
                float ypos = h / (float)(topheight-1) * (float)(bottomheight-1);
                
                // Get interpolated sample
                float tlx = floor(xpos);
                float tly = floor(ypos);
                int srcIdxOff = bottomwidth*(bottomheight*cn + tly) + tlx;
                
                float sampleTL = bottom_data[srcIdxOff];
                float sampleTR = bottom_data[min(srcIdxOff+1,bottomcount)];
                float sampleBL = bottom_data[min(srcIdxOff+bottomwidth,bottomcount)];
                float sampleBR = bottom_data[min(srcIdxOff+1+bottomwidth,bottomcount)];
                
                float xdist = xpos - tlx;
                float ydist = ypos - tly;
                
                float sample = (1-xdist)*(1-ydist)*sampleTL
                            + (  xdist)*(  ydist)*sampleBR
                            + (1-xdist)*(  ydist)*sampleBL
                            + (  xdist)*(1-ydist)*sampleTR;
                            
                int destindex = w + (topwidth * h) + (topwidth*topheight*c) + top_numstride*n;
                top_data_chanoffsetptr[destindex] = sample;
            }
        }
    }
    }
    
    cntChannelsOfPrevious += bottom[i]->channels();
  }
  
  CHECK_EQ(cntChannelsOfPrevious, top[0]->channels()); //Should have processed all top channels
  
  return;
}

template <typename Dtype>
void AccumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  
  int topwidth = top[0]->width();
  int topheight = top[0]->height();
  int topchannels = top[0]->channels();
  
  int nbottom=bottom.size();
  if(this->layer_param_.accum_param().have_reference())
      nbottom--;

  int cntChannelsOfPrevious = 0;
  for (int i = 0; i < nbottom; ++i) {
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    
    int bottomcount = bottom[i]->count();
    
    memset(bottom_diff, 0, sizeof(Dtype) * bottomcount);
    
    int bottomnum = bottom[i]->num();
    int bottomchannels = bottom[i]->channels();
    int bottomwidth = bottom[i]->width();
    int bottomheight = bottom[i]->height();
    
    const Dtype* top_diff_chanoffsetptr = top_diff + (top[0]->offset(0, cntChannelsOfPrevious));
    
    //int topcount = topwidth * topheight * bottomchannels * bottomnum;
    
    // Number of elements in toplayer sub-part corresponding to current bottom[i]
    //int top_countpernum = topwidth * topheight * bottomchannels;
    
    // Number of elements in one toplayer disregarding num. This is the amount of entries to skip from one num to the next
    int top_numstride = topwidth * topheight * topchannels;
    
    for(int n = 0; n < bottomnum; n++) {
    for(int c = 0; c < bottomchannels; c++) {
        int cn = n*bottomchannels + c;
        for(int h = 0; h < topheight; h++) {
            for(int w = 0; w < topwidth; w++) {
                            
                float xpos = w / (float)(topwidth-1) * (float)(bottomwidth-1);
                float ypos = h / (float)(topheight-1) * (float)(bottomheight-1);
                
                // Get interpolated sample
                float tlx = floor(xpos);
                float tly = floor(ypos);
                int srcIdxOff = bottomwidth*(bottomheight*cn + tly) + tlx;
                
                float xdist = xpos - tlx;
                float ydist = ypos - tly;
                            
                int topindex = w + (topwidth * h) + (topwidth*topheight*c) + top_numstride*n;

                float topdiffsample = top_diff_chanoffsetptr[topindex];
                
                bottom_diff[srcIdxOff] += (1-xdist)*(1-ydist) * topdiffsample;
                if(tlx+1 < bottomwidth) bottom_diff[min(srcIdxOff+1,bottomcount)] += (xdist)*(1-ydist) * topdiffsample;
                if(tly+1 < bottomheight) bottom_diff[min(srcIdxOff+bottomwidth,bottomcount)] += (1-xdist)*(ydist) * topdiffsample;
                if(tlx+1 < bottomwidth && tly+1 < bottomheight) bottom_diff[min(srcIdxOff+1+bottomwidth,bottomcount)] += (xdist)*(ydist) * topdiffsample;
                
                //top_data_chanoffsetptr[destindex] = sample;
                
            }
        }
    }
    }
    
    cntChannelsOfPrevious += bottom[i]->channels();
  }
  
  CHECK_EQ(cntChannelsOfPrevious, top[0]->channels()); //Should have processed all top channels
  
}

#ifdef CPU_ONLY
STUB_GPU(AccumLayer);
#endif

INSTANTIATE_CLASS(AccumLayer);
REGISTER_LAYER_CLASS(Accum);


}  // namespace caffe

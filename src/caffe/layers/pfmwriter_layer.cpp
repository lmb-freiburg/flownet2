// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/pfm_writer_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output.hpp"
#include "caffe/net.hpp"
#include "caffe/solver.hpp"

#include "thirdparty/CImg/CImg.h"
using namespace cimg_library;

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>
#include <sys/dir.h>

using std::max;

namespace caffe {
  
template <typename Dtype>
void PFMWriterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{


}

template <typename Dtype>
void PFMWriterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    DIR* dir = opendir(this->layer_param_.writer_param().folder().c_str());
    if (dir)
        closedir(dir);
    else if (ENOENT == errno) {
        std::string cmd("mkdir -p " + this->layer_param_.writer_param().folder());
        int retval = system(cmd.c_str());
        (void)retval;
    }

    CHECK_EQ(bottom.size(), 1) << "PFMWRITER layer takes one input";
}

/**
 * @brief Write float data to a single- or triple-channel PFM file
 *
 *        PLEASE NOTE that this function NEGATES all values because it
 *        was written to work with a DispNet network which internally
 *        represents disparities as offsets in x-direction, and so a
 *        left-view-to-right-view disparity map is all negative.
 *
 * @param filename Target filepath of the PFM output
 * @param data float* data to be written
 * @param xSize Width of the image
 * @param ySize Height of the image
 * @param channels Depth of the image (must be 1 or 3)
 */
void writePFMFile(const std::string& filename,
                  const float* data,
                  int xSize,
                  int ySize,
                  int channels)
{
  CImg<float> tmp_img(xSize,ySize,1,channels);
  cimg_forXYC(tmp_img,x,y,c){
    tmp_img(x,y,0,c) = -1.f*data[c*xSize*ySize + y*xSize + x];
  }
  tmp_img.save_pfm(filename.c_str());
}

template <typename Dtype>
void PFMWriterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    
    Net<Dtype> *net = this->GetNet();
    int iter = net->iter();

    int size=height*width*channels;
    for(int n=0; n<num; n++)
    {
        char filename[256];
        if(this->layer_param_.writer_param().has_file())
            strcpy(filename,this->layer_param_.writer_param().file().c_str());
        else
        {
            if(num>1)
                sprintf(filename,"%s/%s%07d(%03d)%s.pfm",
                    this->layer_param_.writer_param().folder().c_str(),
                    this->layer_param_.writer_param().prefix().c_str(),
                    iter,
                    n,
                    this->layer_param_.writer_param().suffix().c_str()
                );
            else
                sprintf(filename,"%s/%s%07d%s.pfm",
                    this->layer_param_.writer_param().folder().c_str(),
                    this->layer_param_.writer_param().prefix().c_str(),
                    iter,
                    this->layer_param_.writer_param().suffix().c_str()
                );
        }

        const Dtype* data=bottom[0]->cpu_data()+n*size;

        LOG(INFO) << "Saving " << filename;
        writePFMFile(filename,(const float*)data,width,height,channels);
    }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(PFMWriterLayer, Forward);
#endif

INSTANTIATE_CLASS(PFMWriterLayer);
REGISTER_LAYER_CLASS(PFMWriter);


}  // namespace caffe

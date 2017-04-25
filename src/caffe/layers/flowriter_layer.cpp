// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/flo_writer_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output.hpp"
#include "caffe/net.hpp"
#include "caffe/solver.hpp"

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
void FLOWriterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{


}

template <typename Dtype>
void FLOWriterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(bottom.size(), 1) << "FLOWRITER layer takes one input";

    //const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    //const int height = bottom[0]->height();
    //const int width = bottom[0]->width();

    CHECK_EQ(channels, 2) << "FLOWRITER layer input must have two channels";

    DIR* dir = opendir(this->layer_param_.writer_param().folder().c_str());
    if (dir)
        closedir(dir);
    else if (ENOENT == errno) {
        std::string cmd("mkdir -p " + this->layer_param_.writer_param().folder());
        int retval = std::system(cmd.c_str());
        (void)retval;
    }
}

template <typename Dtype>
void FLOWriterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
                sprintf(filename,"%s/%s%07d(%03d)%s.flo",
                    this->layer_param_.writer_param().folder().c_str(),
                    this->layer_param_.writer_param().prefix().c_str(),
                    iter,
                    n,
                    this->layer_param_.writer_param().suffix().c_str()
                );
            else
                sprintf(filename,"%s/%s%07d%s.flo",
                    this->layer_param_.writer_param().folder().c_str(),
                    this->layer_param_.writer_param().prefix().c_str(),
                    iter,
                    this->layer_param_.writer_param().suffix().c_str()
                );
        }

        const Dtype* data=bottom[0]->cpu_data()+n*size;

        LOG(INFO) << "Saving " << filename;
        writeFloFile(filename,(const float*)data,width,height);
    }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(FLOWriterLayer, Forward);
#endif

INSTANTIATE_CLASS(FLOWriterLayer);
REGISTER_LAYER_CLASS(FLOWriter);

}  // namespace caffe

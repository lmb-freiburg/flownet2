// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/mean_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

#include "caffe/util/io.hpp"

using std::max;

namespace caffe {
  
template <typename Dtype>
void MeanLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void MeanLayer<Dtype>::ReadData()
{
    string mean_file = this->layer_param().mean_param().file();

    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    mean_.FromProto(blob_proto);
}

template <typename Dtype>
void MeanLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_GT(top.size(), 0) << "MEAN needs one output";
    CHECK_EQ(top.size(), bottom.size()) << "MEAN needs as many outputs as inputs";

    if(this->layer_param_.mean_param().has_file())
    {
        ReadData();
        for(int k=0; k<bottom.size(); k++)
        {
            CHECK_EQ(mean_.width(),    bottom[k]->width()) << "mean size does not match bottom size";
            CHECK_EQ(mean_.height(),   bottom[k]->height()) << "mean size does not match bottom size";
            CHECK_EQ(bottom[k]->channels() % mean_.channels(), 0) << "bottom size not divisible by mean size";
        }
    }
    else
    {
        for(int k=0; k<bottom.size(); k++)
            CHECK_EQ(bottom[k]->channels() % this->layer_param_.mean_param().value().size(), 0) << "bottom size not divisible by mean size";
    }

    for(int k=0; k<bottom.size(); k++)
        top[k]->Reshape(bottom[k]->num(),bottom[k]->channels(),bottom[k]->height(),bottom[k]->width());
}

template <typename Dtype>
void MeanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    for(int k=0; k<bottom.size(); k++)
    {
        const Dtype* bottom_data = bottom[k]->cpu_data();
        Dtype* top_data = top[k]->mutable_cpu_data();
        int dim = bottom[k]->count() / bottom[k]->num();
        int num = bottom[k]->num();

        float input_scale = this->layer_param().mean_param().input_scale();
        float mean_scale = this->layer_param().mean_param().mean_scale();
        float output_scale = this->layer_param().mean_param().output_scale();

        if(this->layer_param_.mean_param().has_file())
        {
            const Dtype* mean_data = mean_.cpu_data();

            for (int i=0; i<num; i++)
                for(int j=0; j<dim; j++)
                {
                    Dtype data = bottom_data[i*dim + j];
                    Dtype mean = mean_data[j % mean_.count()];

                    if(this->layer_param_.mean_param().operation() == MeanParameter_MeanOperation_SUBTRACT)
                        mean = -mean;

                    top_data[i*dim + j] = output_scale*(input_scale*data + mean_scale*mean);
                }
        }
        else
        {
            int widthheight=bottom[k]->width()*bottom[k]->height();
            for (int i=0; i<num; i++)
                for(int c=0; c<bottom[k]->channels(); c++)
                    for(int j=0; j<widthheight; j++)
                    {
                        Dtype data = bottom_data[i*dim + c*widthheight + j];
                        Dtype mean = this->layer_param_.mean_param().value().Get(c % this->layer_param_.mean_param().value().size());

                        if(this->layer_param_.mean_param().operation() == MeanParameter_MeanOperation_SUBTRACT)
                            mean = -mean;

                        top_data[i*dim + c*widthheight + j] = output_scale*(input_scale*data + mean_scale*mean);
                    }
        }
    }
}


INSTANTIATE_CLASS(MeanLayer);
REGISTER_LAYER_CLASS(Mean);


}  // namespace caffe

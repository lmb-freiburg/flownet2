#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/black_augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BlackAugmentationKernel(const int n, const Dtype* in,
    Dtype* out, int width, int height, int channels, int border_width, int border_height)
{
  CUDA_KERNEL_LOOP(index, n) {
      int x = index % width;
      int y = (index / width) % height;

      if(x>=border_width && x<width-border_width && y>=border_height && y<height-border_height)
          out[index] = in[index];
      else
          out[index] = 0;
  }
}


template <typename Dtype>
void BlackAugmentationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    // sample coeffs

    int width=bottom[0]->width();
    int height=bottom[0]->height();
    int num=bottom[0]->num();

    int img_size=3*width*height;
    int flow_size=2*width*height;

    for(int n=0; n<num; n++)
    {
        bool black = caffe_rng_generate<Dtype,bool>(this->layer_param_.black_augmentation_param().black(), 1.0, 0.0);
        if(black)
        {
            const Dtype* img_ptr=0;
            for(int i=0; i<top.size(); i++)
                if(top[i]->channels()==3)
                    img_ptr=bottom[i]->gpu_data() + n*img_size;

            for(int i=0; i<top.size(); i++)
            {
                if(top[i]->channels()==2)
                    caffe_gpu_memset(flow_size*sizeof(Dtype), 0, top[i]->mutable_gpu_data()+n*flow_size);
                else if(bottom[i]->gpu_data()+n*img_size!=img_ptr)
                    caffe_gpu_memcpy(img_size*sizeof(Dtype), img_ptr, top[i]->mutable_gpu_data()+n*img_size);
            }

            continue;
        }

        float border = caffe_rng_generate<Dtype,float>(this->layer_param_.black_augmentation_param().border(), 1.0, 0.0);
        if(border!=0.0)
        {
            int border_width = 0;
            int border_height = 0;
            float border_type = rand() % 3;
            if(border_type == 0)
                border_width = border*width;
            else if(border_type==1)
                border_height = border*height;
            else
            {
                border_width = border*width;
                border_height = border*height;
            }

            for(int i=0; i<top.size(); i++)
            {
                int count = top[i]->channels()*width*height;
                BlackAugmentationKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count,
                    bottom[i]->gpu_data() + n*count,
                    top[i]->mutable_gpu_data() + n*count,
                    width, height, top[i]->channels(),
                    border_width,
                    border_height
                );
            }

            continue;
        }

        for(int i=0; i<top.size(); i++)
        {
            int count = top[i]->channels()*width*height;
            caffe_gpu_memcpy(width*height*top[i]->channels()*sizeof(Dtype),bottom[i]->gpu_data()+n*count, top[i]->mutable_gpu_data()+n*count);
        }
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(BlackAugmentationLayer);

}  // namespace caffe

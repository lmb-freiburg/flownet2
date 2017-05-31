#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/black_augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BlackAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK(bottom.size() == top.size()) << "Number of bottom blobs must match number of top blobs";

    int width = bottom[0]->width();
    int height = bottom[0]->height();
    int num = bottom[0]->num();

    for(int i=0; i<bottom.size(); i++)
    {
        CHECK(bottom[i]->channels()==2 || bottom[i]->channels()==3) << "Inputs must have 2 or 3 channels";
        CHECK(bottom[i]->num() == num) << "All blobs must have same num dimension";
        CHECK(bottom[i]->height() == height) << "All blobs must have same height dimension";
        CHECK(bottom[i]->width() == width) << "All blobs must have same width dimension";
    }
}

template <typename Dtype>
void BlackAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    for(int i=0; i<bottom.size(); i++)
        top[i]->ReshapeLike(*bottom[i]);
}

template <typename Dtype>
void BlackAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(BlackAugmentationLayer);
#endif

INSTANTIATE_CLASS(BlackAugmentationLayer);
REGISTER_LAYER_CLASS(BlackAugmentation);

}  // namespace caffe

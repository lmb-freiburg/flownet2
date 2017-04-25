// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/custom_data_layer.hpp"
#include "caffe/util/io.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void CustomDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    Forward_cpu(bottom,top);
}

INSTANTIATE_LAYER_GPU_FUNCS(CustomDataLayer);

}  // namespace caffe

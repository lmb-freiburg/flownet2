#ifndef PHIL_DATA_LAYER_HPP_
#define PHIL_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "lmdb.h"
#include "caffe/util/db.hpp"

namespace caffe {


/**
 * @brief Customs Data Layer
 *
 */
template <typename Dtype>
void* CustomDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class CustomDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* CustomDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit CustomDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~CustomDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline bool ShareInParallel() const { return true; }
  
  virtual inline const char* type() const { return "CustomData"; }
  
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();
  
  virtual void generateRandomPermutation(int seed, int block_size = 0);

  shared_ptr<Caffe::RNG> prefetch_rng_;

  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

  int database_entries_;
  int datum_index_;
  int range_start_;
  int range_end_;
  int range_size_;
  
  int preselection_label_;
  
  std::vector<int> permutation_vector_;
  
  vector<int> slice_point_;
  vector<int> channel_encoding_;
  
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  //shared_ptr<Blob<Dtype> > prefetch_data_;
  vector<shared_ptr<Blob<Dtype> > > prefetch_data_blobs_;
  
  Blob<Dtype> data_mean_;
  bool output_labels_;
  
  int iter_;
};


}  // namespace caffe

#endif  

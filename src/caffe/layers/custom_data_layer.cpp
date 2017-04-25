// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>
#include <boost/algorithm/string.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/custom_data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

#include "lmdb.h"

using std::string;

namespace caffe {

const int kMaxKeyLength = 256;

template <typename Dtype>
void CustomDataLayer<Dtype>::generateRandomPermutation(int seed, int block_size) {
  if (seed > 0)
    std::srand (unsigned(seed));
  if (block_size > 0) {
    int num_blocks = (permutation_vector_.size() + block_size - 1) / block_size; // equal to ceil(size / block_size)
    for (int b=0; b < num_blocks; ++b) {
      int n1 = b * block_size;
      int n2 = std::min((b+1)*block_size, static_cast<int>(permutation_vector_.size()));
      std::random_shuffle(permutation_vector_.begin() + n1, permutation_vector_.begin() + n2 -1);
    }
  } else {
    std::random_shuffle(permutation_vector_.begin(), permutation_vector_.end());
  }
}

template <typename Dtype>
void DecodeData(Dtype*& ptr,Datum& datum,const vector<int>& slice_points,const vector<int>& encoding)
{
    int width=datum.width();
    int height=datum.height();
    int channels=datum.channels();
    int count=width*height*channels;

    ptr=new Dtype[count];
    if(datum.float_data_size())
    {
        CHECK_EQ(encoding.size(),0) << "Encoded layers must be stored as uint8 in LMDB.";

        for(int i=0; i<count; i++)
            ptr[i]=datum.float_data(i);

        return;
    }

    const unsigned char* srcptr=(const unsigned char*)datum.data().c_str();
    Dtype* destptr=ptr;

    int channel_start = -1; //inclusive
    int channel_end = 0; //non-inclusive (end will become start in next slice)
    for(int slice = 0; slice <= slice_points.size(); slice++)
    {
        channel_start = channel_end;

        if(slice == slice_points.size())
            channel_end = channels;
        else
            channel_end = slice_points[slice];

        int channel_count=channel_end-channel_start;

        int format;
        if(encoding.size()<=slice)
            format=DataParameter_CHANNELENCODING_UINT8;
        else
            format=encoding[slice];

//         LOG(INFO) << "Slice " << slice << "(" << channel_start << "," << channel_end << ") has format " << ((int)format);
        switch(format)
        {
            case DataParameter_CHANNELENCODING_UINT8:
                for(int c=0; c<channel_count; c++)
                    for(int y=0; y<height; y++)
                        for(int x=0; x<width; x++)
                            *(destptr++)=static_cast<Dtype>(*(srcptr++));
                break;
            case DataParameter_CHANNELENCODING_UINT16FLOW:
            for(int c=0; c<channel_count; c++)
                for(int y=0; y<height; y++)
                    for(int x=0; x<width; x++)
                    {
                        short v;
                        *((unsigned char*)&v)=*(srcptr++);
                        *((unsigned char*)&v+1)=*(srcptr++);

                        Dtype value;
                        if(v==std::numeric_limits<short>::max()) {
                          value = std::numeric_limits<Dtype>::signaling_NaN();
                        } else {
                          value = ((Dtype)v)/32.0;
                        }

                        *(destptr++)=value;
                    }
                break;
            case DataParameter_CHANNELENCODING_BOOL1:
                {
                    int j=0;
                    for(int i=0; i<(width*height-1)/8+1; i++)
                    {
                        unsigned char data=*(srcptr++);
                        for(int k=0; k<8; k++)
                        {
                            float value=(data&(1<<k))==(1<<k);
                            if(j<width*height)
                                *(destptr++)=value?1.0:0;
                            j++;
                        }
                    }
                }
                break;
            default:
                LOG(FATAL) << "Invalid format for slice " << slice;
                break;
        }
    }
//     LOG(INFO) << destptr << " " << ptr;
    assert(destptr==ptr+count);
}

template <typename Dtype>
void* CustomDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  CustomDataLayer<Dtype>* layer = static_cast<CustomDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  Datum datum;
  Dtype* top_label = NULL;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
  }
  const Dtype scale = layer->layer_param_.data_param().scale();
  const int batch_size = layer->layer_param_.data_param().batch_size();
  const int crop_size = layer->layer_param_.data_param().crop_size();
  const bool mirror = layer->layer_param_.data_param().mirror();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  //const int size = layer->datum_size_;
  
  const int heightwidth = height*width;
  
  const Dtype* mean = layer->data_mean_.cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // go to the next iter
    switch (layer->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      LOG(FATAL) << "LevelDB not supported by CustomData";
      break;
    case DataParameter_DB_LMDB: {
      if(layer->datum_index_ >= layer->range_size_) layer->datum_index_ = 0; //wrap around
      
      int dbIndex = layer->permutation_vector_.at(layer->datum_index_); // optionally shifted and permuted position (range and rand_perm)
      
      //LOG(INFO) << "Fetching: datum " << layer->datum_index_ << "/" << layer->range_size_ << ". I.e. permuted&ranged: " << dbIndex << ". ";
      
      char key_cstr[kMaxKeyLength];
      snprintf(key_cstr, kMaxKeyLength, "%08d", dbIndex);
      layer->mdb_key_.mv_data = (void*)key_cstr;
      layer->mdb_key_.mv_size = strlen((char*)layer->mdb_key_.mv_data);
      if(mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_, &layer->mdb_value_, MDB_SET_RANGE) != MDB_SUCCESS) {
        LOG(FATAL) << "Internal data fetch error: Tried to fetch element " << layer->datum_index_ << " of " << layer->range_size_ << " which is in DB: " << dbIndex;
      }
      
      layer->datum_index_++;
      
    } break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // get a blob
    switch (layer->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      LOG(FATAL) << "LevelDB not supported by CustomData";
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
              &layer->mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(layer->mdb_value_.mv_data,
          layer->mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // Iterate over slices:
    int src_channel_start = -1; //inclusive
    int src_channel_end = 0; //non-inclusive (end will become start in next slice)
    
    Dtype* decoded_data;
    DecodeData(decoded_data,datum,layer->slice_point_,layer->channel_encoding_); // Can deal with all types of data

    for(int slice = 0; slice <= layer->slice_point_.size(); slice++) {
        src_channel_start = src_channel_end;

        if(slice == layer->slice_point_.size()) { // last slice
            src_channel_end = channels;
        } else {
            src_channel_end = layer->slice_point_[slice];
        }
        
        CHECK(layer->prefetch_data_blobs_[slice]);
        Dtype* top_data = layer->prefetch_data_blobs_[slice]->mutable_cpu_data();

        int slice_channel_count = src_channel_end - src_channel_start;

        if (crop_size) {
            int h_off, w_off;
            // We only do random crop when we do training.
            /*if (layer->phase_ == Caffe::TRAIN) {
          h_off = layer->PrefetchRand() % (height - crop_size);
          w_off = layer->PrefetchRand() % (width - crop_size);
          } else {*/
            // Always do fixed crop (also when we do training)
            h_off = (height - crop_size) / 2;
            w_off = (width - crop_size) / 2;
            //}
            if (mirror && layer->PrefetchRand() % 2) {
                // Copy mirrored version
                for (int c = 0; c < slice_channel_count; ++c) {
                    for (int h = 0; h < crop_size; ++h) {
                        for (int w = 0; w < crop_size; ++w) {
                            int srcc = src_channel_start + c;

                            int top_index = ((item_id * slice_channel_count + c) * crop_size + h)
                                    * crop_size + (crop_size - 1 - w);
                            int data_index = (srcc * height + h + h_off) * width + w + w_off;
                            Dtype datum_element = decoded_data[data_index];
                            top_data[top_index] = (datum_element - mean[data_index]) * scale;
                        }
                    }
                }
            } else {
                // Normal copy
                for (int c = 0; c < slice_channel_count; ++c) {
                    for (int h = 0; h < crop_size; ++h) {
                        for (int w = 0; w < crop_size; ++w) {
                            int srcc = src_channel_start + c;

                            int top_index = ((item_id * slice_channel_count + c) * crop_size + h)
                                    * crop_size + w;
                            int data_index = (srcc * height + h + h_off) * width + w + w_off;
                            Dtype datum_element = decoded_data[data_index];
                            top_data[top_index] = (datum_element - mean[data_index]) * scale;
                        }
                    }
                }
            }
        } else {
            // we will prefer to use data() first if existant, and then try float_data()
            for (int c = 0; c < slice_channel_count; ++c) {
                for (int hw = 0; hw < heightwidth; ++hw) {
                    int srcc = src_channel_start + c;

                    int top_index = (item_id * slice_channel_count + c) * heightwidth + hw;
                    int data_index = srcc * heightwidth + hw;

                    Dtype datum_element = decoded_data[data_index];
                    top_data[top_index] = (datum_element - mean[data_index]) * scale;
                }
            }
        }

        //DEBUG:
        /*std::stringstream sstm;
      sstm << "debugout." << layer->layer_param_.name() << "." << item_id << "." << layer->mdb_cursor_ << ".txt";
      std::ofstream outfile(sstm.str().c_str());
      for (int j = 0; j < size; ++j) {
            outfile << top_data[item_id * size + j] << std::endl;
      }*/


        if (layer->output_labels_) {
            top_label[item_id] = datum.label();
        }
    } //end-for slice iterator
    free(decoded_data);

  } //end-for batch item

  return static_cast<void*>(NULL);
}

template <typename Dtype>
CustomDataLayer<Dtype>::~CustomDataLayer<Dtype>() {
  JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void CustomDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const DataParameter& data_param = this->layer_param_.data_param();
  slice_point_.clear();
  std::copy(data_param.slice_point().begin(),
      data_param.slice_point().end(),
      std::back_inserter(slice_point_));
  std::copy(data_param.encoding().begin(),
      data_param.encoding().end(),
      std::back_inserter(channel_encoding_));
  iter_ = 0;

        
  const bool verbose = this->layer_param_.data_param().verbose();
  if (top.size() == slice_point_.size() + 1) {
    output_labels_ = false;
  } else if (top.size() == slice_point_.size() + 2) {
    output_labels_ = true;
  } else {
    LOG(FATAL) << "CustomDataLayer has " << top.size() << " top blobs, but " << (slice_point_.size()+1) << " slices."; 
  }
  
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
      LOG(FATAL) << "LevelDB not supported by CustomData";
    }
    break;
  case DataParameter_DB_LMDB:
    {
        CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
        CHECK_EQ(mdb_env_open(mdb_env_,
                 this->layer_param_.data_param().source().c_str(),
                 MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
        int dead;
        CHECK_EQ(mdb_reader_check(mdb_env_, &dead), MDB_SUCCESS) << "mdb_reader_check failed";
        LOG(INFO) << "LMDB: Cleaned " << dead << " stale readers.";
        CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
            << "mdb_open failed";
        CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
            << "mdb_cursor_open failed";

        if (verbose)
          LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();

        MDB_stat stat;
        CHECK_EQ(mdb_stat(mdb_txn_, mdb_dbi_, &stat), MDB_SUCCESS) << "mdb_stat failed";


        LOG(INFO) << "DB entries: " << stat.ms_entries;
        database_entries_ = stat.ms_entries;

        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
            MDB_SUCCESS) << "mdb_cursor_get failed";
        datum_index_ = 0;
    }
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
  
  if (verbose && this->layer_param_.data_param().block_size())
    LOG(INFO) << "Block size: " << this->layer_param_.data_param().block_size();

  bool doPreselection = this->layer_param_.data_param().has_preselection_file();
  
  // Preselection
  if (doPreselection) {
    CHECK(this->layer_param_.data_param().has_preselection_label()) << "Preselection file without specifying preselection_label doesn't make sense.";
    
    preselection_label_ = this->layer_param_.data_param().preselection_label();
    
    // Todo: make vector (load from file);
    std::ifstream infile(this->layer_param_.data_param().preselection_file().c_str());
    //std::vector<std::pair<string, int> > lines;
    int label;
    permutation_vector_.clear();
    int index = 0;
    while (infile >> label) {
      // Add samples with correct label
      if(label == preselection_label_) permutation_vector_.push_back(index);
      index++;
    }
    
    CHECK_EQ(database_entries_, index) << "Preselection error: Database has " << database_entries_ << " entries and preselection file contains " << index << " labels.";
    LOG(INFO) << "Preselection: Selected " << permutation_vector_.size() << " of " << database_entries_ << " entries.";
  } else {
    // Fill permutation_vector_ with all entries
    permutation_vector_.clear();
    for (int i=0; i<database_entries_; ++i) permutation_vector_.push_back(i); // 0 1 2 3 4 5 6 7 8 9 ... N
    CHECK_EQ(database_entries_, permutation_vector_.size()) << "Internal range error";
  }
  
  
  // Range settings
  range_start_ = this->layer_param_.data_param().range_start();
  range_end_ = this->layer_param_.data_param().range_end();
  
  if(range_start_ < 0) range_start_ = 0;
  if(range_start_ >= permutation_vector_.size()) range_start_ = permutation_vector_.size() - 1;
  
  if(range_end_ < 0 || range_end_ >= permutation_vector_.size()) range_end_ = permutation_vector_.size() - 1;
  if(range_end_ < range_start_) LOG(FATAL) << "Range end is before start.";
         
  range_size_ = range_end_ - range_start_ + 1;
  
  if (verbose)
    LOG(INFO) << "Data range: " << range_start_ << " to " << range_end_ << " = " << range_size_ << " entries.";

  switch (this->layer_param_.data_param().rand_permute_order()) {
  case DataParameter_RANDPERMORDER_FIRST_PERMUTE_THEN_RANGE:
    // Optionally permute
    if (this->layer_param_.data_param().rand_permute()) {
      int randseed = this->layer_param_.data_param().rand_permute_seed();    
      generateRandomPermutation(randseed, this->layer_param_.data_param().block_size());
    }
    
    // Cut out range:
    if(range_end_ < permutation_vector_.size()-1) {
      permutation_vector_.erase(permutation_vector_.begin()+range_end_+1,permutation_vector_.end());
    }
    if(range_start_ > 0) {
      permutation_vector_.erase(permutation_vector_.begin(),permutation_vector_.begin()+range_start_);
    }
    
    break;
  case DataParameter_RANDPERMORDER_FIRST_RANGE_THEN_PERMUTE:
    // Cut out range of potentially preselected entries:
    if(range_end_ < permutation_vector_.size()-1) {
      permutation_vector_.erase(permutation_vector_.begin()+range_end_+1,permutation_vector_.end());
    }
    if(range_start_ > 0) {
      permutation_vector_.erase(permutation_vector_.begin(),permutation_vector_.begin()+range_start_);
    }
    
    // Optionally permute
    if (this->layer_param_.data_param().rand_permute()) {
      int randseed = this->layer_param_.data_param().rand_permute_seed();    
      generateRandomPermutation(randseed, this->layer_param_.data_param().block_size());
    }
      
    break;
  default:
    LOG(FATAL) << "Unknown rand";
  }
  
  CHECK_EQ(range_size_, permutation_vector_.size()) << "Internal range error";
  if (verbose) {
    printf("Permutation:\n");
    for(int j = 0; j < permutation_vector_.size(); j++) {
      printf("%d ",permutation_vector_.at(j));
    }
    printf("\n");
  }
  
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
        LOG(FATAL) << "No rand_skip for CustomData layer";
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    LOG(FATAL) << "LevelDB not supported by CustomData";
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.data_param().crop_size();
  if (crop_size > 0) {
    LOG(FATAL) << "Cropping currently not supported";
  }

  // Clear all prefetch blobs
  prefetch_data_blobs_.clear();
  
  // Top Blobs may be split:
  if(slice_point_.size() > 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), datum.channels());
    int count = 0;
    int prev = 0;
    vector<int> slices;
    for (int i = 0; i < slice_point_.size(); ++i) {
      CHECK_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }
    slices.push_back(datum.channels() - prev);
    
    for (int i = 0; i < top.size(); ++i) {
      // Prefetch blob:
      shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>(this->layer_param_.data_param().batch_size(),
          slices[i],
          datum.height(), datum.width()
        ));
      
      prefetch_data_blobs_.push_back(blob_pointer);
      
      // Top blob:
      top[i]->Reshape(this->layer_param_.data_param().batch_size(),
                         slices[i],
                         datum.height(), datum.width());
      count += top[i]->count();
      
      if (verbose)
        LOG(INFO) << "output " << i << " data size: " << top[i]->num() << ","
        << top[i]->channels() << "," << top[i]->height() << ","
        << top[i]->width();
    }
    CHECK_EQ(count, this->layer_param_.data_param().batch_size() * datum.channels() * datum.height() * datum.width());
  } else {
    // Prefetch blob:
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width()
      ));
    
    prefetch_data_blobs_.push_back(blob_pointer);
    
    // Top blob:
    top[0]->Reshape(
      this->layer_param_.data_param().batch_size(), datum.channels(),
      datum.height(), datum.width());
      LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  }
    

  
  // label
  if (output_labels_) {
    int label_topblob = slice_point_.size() + 1;
    top[label_topblob]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    prefetch_label_.reset(new Blob<Dtype>(this->layer_param_.data_param().batch_size(), 1, 1, 1));
  }
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(datum_height_, crop_size);
  CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else if (this->layer_param_.data_param().subtract().size()) {
      data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);

      if(this->layer_param_.data_param().subtract().size())
      {
          int means=this->layer_param_.data_param().subtract().size();

          int j=0;
          for(int i=0; i<means; i++)
          {
              Dtype mean=this->layer_param_.data_param().subtract().Get(i);
              LOG(INFO) << "Subtracting " << mean << " from channel " << i;

              for(int i=0; i<data_mean_.width()*data_mean_.height(); i++)
                  data_mean_.mutable_cpu_data()[j++]=mean;
          }
      }
      else
      {
          LOG(FATAL) << "need one or three values to subtract";
      }

  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }


  
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  for (int i = 0; i <= slice_point_.size(); ++i) {
    prefetch_data_blobs_[i]->mutable_cpu_data();
  }
  
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}


template <typename Dtype>
void CustomDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void CustomDataLayer<Dtype>::CreatePrefetchThread() {
  const bool prefetch_needs_rand = (this->phase_ == TRAIN) &&
      (this->layer_param_.data_param().mirror() ||
       this->layer_param_.data_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, CustomDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void CustomDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int CustomDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void CustomDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  // First, join the thread
  JoinPrefetchThread();
  
  for (int i = 0; i <= slice_point_.size(); ++i) {
    // Copy the data
    caffe_copy(prefetch_data_blobs_[i]->count(), prefetch_data_blobs_[i]->cpu_data(),  top[i]->mutable_cpu_data());
    
  }
  
  if (output_labels_) {
    int label_topblob = slice_point_.size() + 1;
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(), top[label_topblob]->mutable_cpu_data());
  }
  
  iter_++;
  if (this->layer_param_.data_param().rand_permute() && this->layer_param_.data_param().permute_every_iter()) {
    if (iter_ % this->layer_param_.data_param().permute_every_iter() == 0) {
      generateRandomPermutation(-1, this->layer_param_.data_param().block_size());
      if (this->layer_param_.data_param().verbose()) {
        printf("Re-permuting at iteration %d. Permutation:\n", iter_);
        for(int j = 0; j < permutation_vector_.size(); j++) {
          printf("%d ",permutation_vector_.at(j));
        }
        printf("\n");
      }
    }
  }
    
  
  // Start a new prefetch thread
  CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(CustomDataLayer, Forward);
#endif

INSTANTIATE_CLASS(CustomDataLayer);
REGISTER_LAYER_CLASS(CustomData);

}  // namespace caffe

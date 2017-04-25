#ifndef PHIL_DATA_AUGMENTATION_LAYER_HPP_
#define PHIL_DATA_AUGMENTATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/augmentation_layer_base.hpp"

namespace caffe {

/**
 * @brief Phil Data Augmentation Layer
 *
 */
template <typename Dtype>
class DataAugmentationLayer : public AugmentationLayerBase<Dtype>, public Layer<Dtype> {
public:
    explicit DataAugmentationLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual ~DataAugmentationLayer() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    virtual void adjust_blobs(vector<Blob<Dtype>*> blobs);
    virtual inline bool AllowBackward() const { LOG(WARNING) << "DataAugmentationLayer does not do backward."; return false; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { for(int i=0; i<propagate_down.size(); i++) if(propagate_down[i]) LOG(FATAL) << "DataAugmentationLayer cannot do backward."; return; }
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { for(int i=0; i<propagate_down.size(); i++) if(propagate_down[i]) LOG(FATAL) << "DataAugmentationLayer cannot do backward."; return; }
              
    
    virtual inline bool DoesUseCustomCopyBlobs() const { return true; }
    virtual inline void CustomCopyBlobs(vector<Blob<Dtype>*> blobs) {
        adjust_blobs(blobs);
    }
    
    int cropped_height_;
    int cropped_width_;
    bool do_cropping_;
    int num_params_;

    bool output_params_;
    bool input_params_;
    Blob<Dtype> all_coeffs_;
    shared_ptr<SyncedMemory> coeff_matrices_;
    shared_ptr<SyncedMemory> coeff_chromatic_;
    shared_ptr<SyncedMemory> coeff_chromatic_eigen_;
    shared_ptr<SyncedMemory> coeff_effect_;
    shared_ptr<SyncedMemory> coeff_colors_;
    shared_ptr<SyncedMemory> chromatic_eigenspace_;

    shared_ptr<SyncedMemory> noise_;

    AugmentationParameter aug_;
    CoeffScheduleParameter discount_coeff_schedule_;
    Blob<Dtype> ones_;
    Blob<Dtype> pixel_rgb_mean_from_proto_;
};

}  // namespace caffe

#endif  // PHIL_DATA_AUGMENTATION_LAYER_HPP_

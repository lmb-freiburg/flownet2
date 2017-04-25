// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/generate_augmentation_parameters_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <fstream>

#define CUDART_NAN_F            __int_as_float(0x7fffffff)

namespace caffe {

template <typename Dtype>
void GenerateAugmentationParametersLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        
  // From bottom to top
  num_iter_++;
  
  Dtype* out_params = (top)[0]->mutable_cpu_data();
  
  Dtype discount_coeff = discount_coeff_schedule_.initial_coeff() + 
      (discount_coeff_schedule_.final_coeff() - discount_coeff_schedule_.initial_coeff()) * (Dtype(2) /
      (Dtype(1) + exp((Dtype)-1.0986 * num_iter_ / discount_coeff_schedule_.half_life())) - Dtype(1));
      
//   LOG(INFO) << "num_iter=" << num_iter_ << ", discount_coeff=" << discount_coeff;
  
    //   We only do transformations during training or if specifically asked to do them during testing.
  bool gen_spatial_transform   = false;
  bool gen_chromatic_transform = false;
  bool gen_effect_transform    = false;
  bool gen_chromatic_eigen_transform = false;
  if(this->phase_ == TRAIN || aug_.augment_during_test()) {
      if(aug_.has_mirror() || aug_.has_rotate() || aug_.has_zoom() || aug_.has_translate() || aug_.has_squeeze() || aug_.has_translate_x() || aug_.has_translate_y())
          gen_spatial_transform   = true;
      if(aug_.has_brightness() || aug_.has_gamma() || aug_.has_contrast() || aug_.has_color())
          gen_chromatic_transform = true;
      if(aug_.has_fog_size() || aug_.has_fog_amount() || aug_.has_motion_blur_angle() || aug_.has_motion_blur_size() || aug_.has_shadow_angle() ||
         aug_.has_shadow_distance() || aug_.has_shadow_strength() || aug_.has_noise() )
          gen_effect_transform = true;
      if(aug_.has_lmult_pow() || aug_.has_lmult_mult() || aug_.has_lmult_add() || aug_.has_sat_pow() || aug_.has_sat_mult() || 
         aug_.has_sat_add() || aug_.has_col_pow() || aug_.has_col_mult() || aug_.has_col_add() || aug_.has_ladd_pow() || 
         aug_.has_ladd_mult() || aug_.has_ladd_add() || aug_.has_col_rotate() )
          gen_chromatic_eigen_transform = true;
  }   
  
  if (gen_spatial_transform) {
    CHECK_GE(cropped_width_, 1) << "Must provide cropped_width with a bottom blob or in the prototxt to do spatial augmentations";
    CHECK_GE(cropped_height_, 1) << "Must provide cropped_height with a bottom blob or in the prototxt to do spatial augmentations";
    CHECK_GE(bottomwidth_, 1) << "Must provide bottomwidth with a bottom blob or in the prototxt to do spatial augmentations";
    CHECK_GE(bottomheight_, 1) << "Must provide bottomheight with a bottom blob or in the prototxt to do spatial augmentations";  
  }
  
  // Preparing the coeffs
  AugmentationCoeff coeff; 
  const Dtype* in_params = bottom[0]->cpu_data();
  
  for (int item_id = 0; item_id < num_; ++item_id) {    
    // Generate spatial coeffs
    if (mode_ == "add" || mode_ == "replace") 
      AugmentationLayerBase<Dtype>::array_to_coeff(in_params + item_id * num_params_, coeff);
    else
      AugmentationLayerBase<Dtype>::clear_all_coeffs(coeff);
    
    if (gen_spatial_transform) {  
      if (mode_ == "replace")
        AugmentationLayerBase<Dtype>::clear_spatial_coeffs(coeff);
      AugmentationLayerBase<Dtype>::generate_valid_spatial_coeffs(aug_, coeff, discount_coeff, bottomwidth_, bottomheight_, 
                                                                  cropped_width_, cropped_height_, 50);
    }
    
    // Write to the output
    AugmentationLayerBase<Dtype>::coeff_to_array(coeff, out_params + item_id * num_params_);
     
    // If also have chromatic transforms, add those
    if (gen_chromatic_transform) {        
      if (mode_ == "regenerate" || mode_ == "replace") {
        AugmentationLayerBase<Dtype>::generate_chromatic_coeffs(aug_, coeff, discount_coeff);
        AugmentationLayerBase<Dtype>::coeff_to_array(coeff, out_params + item_id * num_params_);
      } else {
        AugmentationCoeff tmp_coeff;
        AugmentationLayerBase<Dtype>::generate_chromatic_coeffs(aug_, tmp_coeff, discount_coeff);
        AugmentationLayerBase<Dtype>::add_coeff_to_array(tmp_coeff, out_params + item_id * num_params_);        
      }
    }
    
    // If also have chromatic eigen transforms, add those
    if (gen_chromatic_eigen_transform) {        
      if (mode_ == "regenerate" || mode_ == "replace") {
        AugmentationLayerBase<Dtype>::generate_chromatic_eigen_coeffs(aug_, coeff, discount_coeff);
        AugmentationLayerBase<Dtype>::coeff_to_array(coeff, out_params + item_id * num_params_);
      } else {
        AugmentationCoeff tmp_coeff;
        AugmentationLayerBase<Dtype>::generate_chromatic_eigen_coeffs(aug_, tmp_coeff, discount_coeff);
        AugmentationLayerBase<Dtype>::add_coeff_to_array(tmp_coeff, out_params + item_id * num_params_);        
      }
    }
    
    // If also have effect transforms, add those
    if (gen_effect_transform) {        
      if (mode_ == "regenerate" || mode_ == "replace") {
        AugmentationLayerBase<Dtype>::generate_effect_coeffs(aug_, coeff, discount_coeff);
        AugmentationLayerBase<Dtype>::coeff_to_array(coeff, out_params + item_id * num_params_);
      } else {
        AugmentationCoeff tmp_coeff;
        AugmentationLayerBase<Dtype>::generate_effect_coeffs(aug_, tmp_coeff, discount_coeff);
        AugmentationLayerBase<Dtype>::add_coeff_to_array(tmp_coeff, out_params + item_id * num_params_);        
      }
    } 
    
    
  }
}



INSTANTIATE_LAYER_GPU_FUNCS(GenerateAugmentationParametersLayer);

}  // namespace caffe

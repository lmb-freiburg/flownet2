// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/augmentation_layer_base.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

//============================================ tTransMat functions

template <typename Dtype>
void AugmentationLayerBase<Dtype>::tTransMat::toIdentity()
{
    t0 = 1; t2 = 0; t4 = 0;
    t1 = 0; t3 = 1; t5 = 0;
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::tTransMat::leftMultiply(float u0, float u1, float u2, float u3, float u4, float u5)
{
    float t0 = this->t0, t2 = this->t2, t4 = this->t4;
    float t1 = this->t1, t3 = this->t3, t5 = this->t5;

    this->t0 = t0*u0 + t1*u2;
    this->t1 = t0*u1 + t1*u3;

    this->t2 = t2*u0 + t3*u2;
    this->t3 = t2*u1 + t3*u3;

    this->t4 = t4*u0 + t5*u2 + u4;
    this->t5 = t4*u1 + t5*u3 + u5;
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::tTransMat::fromCoeff(AugmentationCoeff* coeff,int width,int height,int bottomwidth,int bottomheight)
{
    if (coeff->mirror())                            leftMultiply(-1, 0, 0, 1,  .5 * static_cast<float>(width), -.5 * static_cast<float>(height) );
    else                                            leftMultiply( 1, 0, 0, 1, -.5 * static_cast<float>(width), -.5 * static_cast<float>(height) );

    if (coeff->has_angle())                         leftMultiply(cos(coeff->angle()), sin(coeff->angle()), -sin(coeff->angle()), cos(coeff->angle()), 0, 0);
    if (coeff->has_dx() || coeff->has_dy())         leftMultiply(1, 0, 0, 1, coeff->dx() * static_cast<float>(width), coeff->dy() * static_cast<float>(height));
    if (coeff->has_zoom_x() || coeff->has_zoom_y()) leftMultiply(1.0/coeff->zoom_x(), 0, 0, 1.0/coeff->zoom_y(), 0, 0);

    leftMultiply(1, 0, 0, 1, .5 * static_cast<float>(bottomwidth), .5 * static_cast<float>(bottomheight));
}


template <typename Dtype>
typename AugmentationLayerBase<Dtype>::tTransMat AugmentationLayerBase<Dtype>::tTransMat::inverse()
{
    float a = this->t0, c = this->t2, e = this->t4;
    float b = this->t1, d = this->t3, f = this->t5;

    float denom = a*d - b*c;
    
    tTransMat result;
    result.t0 = d / denom;
    result.t1 = -b / denom;
    result.t2 = -c / denom;
    result.t3 = a / denom;
    result.t4 = (c*f-d*e) / denom;
    result.t5 = (b*e-a*f) / denom;
    
    return result;
}

//============================================ Coeff and Array handling

template <typename Dtype>
void AugmentationLayerBase<Dtype>::generate_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff) {    
  if (aug.has_mirror()) 
    coeff.set_mirror(static_cast<float>(caffe_rng_generate<Dtype,bool>(aug.mirror(),coeff.default_instance().mirror())));
  if (aug.has_translate()) {
    coeff.set_dx(caffe_rng_generate<Dtype,float>(aug.translate(), discount_coeff,coeff.default_instance().dx()));
    coeff.set_dy(caffe_rng_generate<Dtype,float>(aug.translate(), discount_coeff,coeff.default_instance().dy()));
  } 
  if (aug.has_translate_x()) {
    coeff.set_dx(caffe_rng_generate<Dtype,float>(aug.translate_x(), discount_coeff,coeff.default_instance().dx()));
  } 
  if (aug.has_translate_y()) {
    coeff.set_dy(caffe_rng_generate<Dtype,float>(aug.translate_y(), discount_coeff,coeff.default_instance().dy()));
  } 
  if (aug.has_rotate())
    coeff.set_angle(caffe_rng_generate<Dtype,float>(aug.rotate(), discount_coeff,coeff.default_instance().angle()));
  if (aug.has_zoom()) {
    coeff.set_zoom_x(caffe_rng_generate<Dtype,float>(aug.zoom(), discount_coeff,coeff.default_instance().zoom_x()));
    coeff.set_zoom_y(coeff.zoom_x());
  }
  if (aug.has_squeeze()) {
    float squeeze_coeff = caffe_rng_generate<Dtype,float>(aug.squeeze(), discount_coeff,1);
    coeff.set_zoom_x(coeff.zoom_x() * squeeze_coeff);
    coeff.set_zoom_y(coeff.zoom_y() / squeeze_coeff);
  }
}

// try to sample parameters for which transformed image doesn't go outside the borders of the original one
// in order to check this, just apply the transformations to 4 corners
template <typename Dtype>
void AugmentationLayerBase<Dtype>::generate_valid_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff,
                                                                 int width, int height, int cropped_width, int cropped_height, int max_num_tries) 
{
  int x, y;
  Dtype x1, y1, x2, y2;
  int counter = 0;
  int good_params = 0;  
  
  int num_params = coeff.GetDescriptor()->field_count();
  Blob<Dtype> in_params_blob(1, num_params, 1, 1);
  Dtype* in_params = in_params_blob.mutable_cpu_data();
  Blob<Dtype> curr_params_blob(1, num_params, 1, 1);
  Dtype* curr_params = curr_params_blob.mutable_cpu_data();  
  
  // convert incoming params to an array
  AugmentationLayerBase<Dtype>::coeff_to_array(coeff, in_params);
    
  while (good_params < 4 && counter < max_num_tries)
  {
    // generate params
    AugmentationLayerBase<Dtype>::clear_all_coeffs(coeff);
    AugmentationLayerBase<Dtype>::generate_spatial_coeffs(aug, coeff, discount_coeff);
//     LOG(INFO) << "DEBUG: try dx = " << coeff.dx() << ", dy = " << coeff.dy();
    
    // add incoming params
    AugmentationLayerBase<Dtype>::coeff_to_array(coeff, curr_params);
    caffe_axpy(num_params, Dtype(1), in_params, curr_params);
    AugmentationLayerBase<Dtype>::array_to_coeff(curr_params, coeff);

    // check if all 4 corners of the transformed image fit into the original image
    good_params = 0;
    for (x = 0; x < cropped_width; x += cropped_width-1)
    {
        for (y = 0; y < cropped_height; y += cropped_height-1)
        {
            // move the origin and mirror
            if (coeff.mirror()) {
                x1 = - static_cast<Dtype>(x) + .5 * static_cast<Dtype>(cropped_width);
                y1 =   static_cast<Dtype>(y) - .5 * static_cast<Dtype>(cropped_height);
            }  else {
                x1 =   static_cast<Dtype>(x) - .5 * static_cast<Dtype>(cropped_width);
                y1 =   static_cast<Dtype>(y) - .5 * static_cast<Dtype>(cropped_height);
            }
            // rotate
            x2 =  cos(coeff.angle()) * x1 - sin(coeff.angle()) * y1;
            y2 =  sin(coeff.angle()) * x1 + cos(coeff.angle()) * y1;
            // translate
            x2 = x2 + coeff.dx() * static_cast<Dtype>(cropped_width);
            y2 = y2 + coeff.dy() * static_cast<Dtype>(cropped_height);
            // zoom
            x2 = x2 / coeff.zoom_x();
            y2 = y2 / coeff.zoom_y();
            // move the origin back
            x2 = x2 + .5 * static_cast<Dtype>(width);
            y2 = y2 + .5 * static_cast<Dtype>(height);

            if (!(floor(x2) < 0 || floor(x2) > static_cast<Dtype>(width - 2) || floor(y2) < 0 || floor(y2) > static_cast<Dtype>(height - 2)))
                good_params++;
        }
    }
    counter++;
  }
  if (counter >= max_num_tries) {
    AugmentationLayerBase<Dtype>::array_to_coeff(in_params, coeff);
    LOG(WARNING) << "Augmentation: Exceeded maximum tries in finding spatial coeffs.";
  }
//   LOG(INFO) << "DEBUG: final dx = " << coeff.dx() << ", dy = " << coeff.dy();
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::add_coeff_to_array(const AugmentationCoeff& coeff, Dtype* out_params) 
{
  int num_params = coeff.GetDescriptor()->field_count();
  Blob<Dtype> curr_params_blob(1, num_params, 1, 1);
  Dtype* curr_params = curr_params_blob.mutable_cpu_data();        
           
  AugmentationLayerBase<Dtype>::coeff_to_array(coeff, curr_params);          
  caffe_axpy(num_params, Dtype(1), curr_params, out_params);
}

//template void AugmentationLayerBase<float>::generate_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, float discount_coeff);
//template void AugmentationLayerBase<double>::generate_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, double discount_coeff);

template <typename Dtype>
void AugmentationLayerBase<Dtype>::clear_spatial_coeffs(AugmentationCoeff& coeff) {    
  coeff.clear_mirror();
  coeff.clear_dx();
  coeff.clear_dy();
  coeff.clear_angle();
  coeff.clear_zoom_x();
  coeff.clear_zoom_y();
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::clear_chromatic_coeffs(AugmentationCoeff& coeff) {
  coeff.clear_gamma();
  coeff.clear_brightness();
  coeff.clear_contrast();
  coeff.clear_color1();
  coeff.clear_color2();
  coeff.clear_color3();
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::clear_chromatic_eigen_coeffs(AugmentationCoeff& coeff) {
  coeff.clear_pow_nomean0();
  coeff.clear_pow_nomean1();
  coeff.clear_pow_nomean2();
  coeff.clear_add_nomean0();
  coeff.clear_add_nomean1();
  coeff.clear_add_nomean2();
  coeff.clear_mult_nomean0();
  coeff.clear_mult_nomean1();
  coeff.clear_mult_nomean2();
  coeff.clear_pow_withmean0();
  coeff.clear_pow_withmean1();
  coeff.clear_pow_withmean2();
  coeff.clear_add_withmean0();
  coeff.clear_add_withmean1();
  coeff.clear_add_withmean2();
  coeff.clear_mult_withmean0();
  coeff.clear_mult_withmean1();
  coeff.clear_mult_withmean2();
  coeff.clear_lmult_pow();
  coeff.clear_lmult_add();
  coeff.clear_lmult_mult();
  coeff.clear_col_angle();
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::clear_effect_coeffs(AugmentationCoeff& coeff) {
  coeff.clear_fog_amount();
  coeff.clear_fog_size();
  coeff.clear_motion_blur_angle();
  coeff.clear_motion_blur_size();
  coeff.clear_shadow_angle();
  coeff.clear_shadow_distance();
  coeff.clear_shadow_strength();
  coeff.clear_noise();
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::clear_all_coeffs(AugmentationCoeff& coeff) {
  clear_spatial_coeffs(coeff);
  clear_chromatic_coeffs(coeff);
  clear_chromatic_eigen_coeffs(coeff);
  clear_effect_coeffs(coeff);
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::generate_chromatic_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff)
{
  if (aug.has_gamma())       coeff.set_gamma(caffe_rng_generate<Dtype,float>(aug.gamma(), discount_coeff));
  if (aug.has_brightness())  coeff.set_brightness(caffe_rng_generate<Dtype,float>(aug.brightness(), discount_coeff));
  if (aug.has_contrast())    coeff.set_contrast(caffe_rng_generate<Dtype,float>(aug.contrast(), discount_coeff));
  if (aug.has_color()) {
    coeff.set_color1(caffe_rng_generate<Dtype,float>(aug.color(), discount_coeff));
    coeff.set_color2(caffe_rng_generate<Dtype,float>(aug.color(), discount_coeff));
    coeff.set_color3(caffe_rng_generate<Dtype,float>(aug.color(), discount_coeff));
  }
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::generate_chromatic_eigen_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff)
{
    if (aug.has_ladd_pow())
        coeff.set_pow_nomean0(caffe_rng_generate<Dtype,float>(aug.ladd_pow(), discount_coeff));
    if (aug.has_col_pow()) {
        coeff.set_pow_nomean1(caffe_rng_generate<Dtype,float>(aug.col_pow(), discount_coeff));
        coeff.set_pow_nomean2(caffe_rng_generate<Dtype,float>(aug.col_pow(), discount_coeff));
    }

    if (aug.has_ladd_add())
        coeff.set_add_nomean0(caffe_rng_generate<Dtype,float>(aug.ladd_add(), discount_coeff));
    if (aug.has_col_add()) {
        coeff.set_add_nomean1(caffe_rng_generate<Dtype,float>(aug.col_add(), discount_coeff));
        coeff.set_add_nomean2(caffe_rng_generate<Dtype,float>(aug.col_add(), discount_coeff));
    }

    if (aug.has_ladd_mult())
        coeff.set_mult_nomean0(caffe_rng_generate<Dtype,float>(aug.ladd_mult(), discount_coeff));
    if (aug.has_col_mult()) {
        coeff.set_mult_nomean1(caffe_rng_generate<Dtype,float>(aug.col_mult(), discount_coeff));
        coeff.set_mult_nomean2(caffe_rng_generate<Dtype,float>(aug.col_mult(), discount_coeff));
    }

    if (aug.has_sat_pow()) {
        coeff.set_pow_withmean1(caffe_rng_generate<Dtype,float>(aug.sat_pow(), discount_coeff));
        coeff.set_pow_withmean2(coeff.pow_withmean1());
    }

    if (aug.has_sat_add()) {
        coeff.set_add_withmean1(caffe_rng_generate<Dtype,float>(aug.sat_add(), discount_coeff));
        coeff.set_add_withmean2(coeff.add_withmean1());
    }

    if (aug.has_sat_mult()) {
        coeff.set_mult_withmean1(caffe_rng_generate<Dtype,float>(aug.sat_mult(), discount_coeff));
        coeff.set_mult_withmean2(coeff.mult_withmean1());
    }

    if (aug.has_lmult_pow())
        coeff.set_lmult_pow(caffe_rng_generate<Dtype,float>(aug.lmult_pow(), discount_coeff));
    if (aug.has_lmult_mult())
        coeff.set_lmult_mult(caffe_rng_generate<Dtype,float>(aug.lmult_mult(), discount_coeff));
    if (aug.has_lmult_add())
        coeff.set_lmult_add(caffe_rng_generate<Dtype,float>(aug.lmult_add(), discount_coeff));
    if (aug.has_col_rotate())
        coeff.set_col_angle(caffe_rng_generate<Dtype,float>(aug.col_rotate(), discount_coeff));
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::generate_effect_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff)
{
    if (aug.has_fog_amount() || aug.has_fog_size())
    {
        coeff.set_fog_amount(caffe_rng_generate<Dtype,float>(aug.fog_amount(), discount_coeff,coeff.default_instance().fog_amount()));
        coeff.set_fog_size(caffe_rng_generate<Dtype,float>(aug.fog_size(), discount_coeff,coeff.default_instance().fog_size()));
    }

    if (aug.has_motion_blur_angle() || aug.has_motion_blur_size())
    {
        coeff.set_motion_blur_angle(caffe_rng_generate<Dtype,float>(aug.motion_blur_angle(), discount_coeff,coeff.default_instance().motion_blur_angle()));
        coeff.set_motion_blur_size(caffe_rng_generate<Dtype,float>(aug.motion_blur_size(), discount_coeff,coeff.default_instance().motion_blur_size()));
    }

    if (aug.has_shadow_angle() || aug.has_shadow_distance() || aug.has_shadow_strength())
    {
        coeff.set_shadow_angle(caffe_rng_generate<Dtype,float>(aug.shadow_angle(), discount_coeff,coeff.default_instance().shadow_angle()));
        coeff.set_shadow_distance(caffe_rng_generate<Dtype,float>(aug.shadow_distance(), discount_coeff,coeff.default_instance().shadow_distance()));
        coeff.set_shadow_strength(caffe_rng_generate<Dtype,float>(aug.shadow_strength(), discount_coeff,coeff.default_instance().shadow_strength()));
    }
    
    if (aug.has_noise())   coeff.set_noise(caffe_rng_generate<Dtype,float>(aug.noise(), discount_coeff));
}

template <typename Dtype>
void AugmentationLayerBase<Dtype>::clear_defaults(AugmentationCoeff& coeff) {
  const google::protobuf::Reflection* ref = coeff.GetReflection();
  const google::protobuf::Descriptor* desc = coeff.GetDescriptor();
  for (int fn = 0 ; fn < desc->field_count(); fn++) {
    const google::protobuf::FieldDescriptor* field_desc = desc->field(fn);
    float field_val = ref->GetFloat(coeff, field_desc);
    if (fabs(field_desc->default_value_float() - field_val) < 1e-3)
      ref->ClearField(&coeff, field_desc);
//     LOG(INFO) << "field " << fn << " cleared";
  }
} 

template <typename Dtype>
void AugmentationLayerBase<Dtype>::coeff_to_array(const AugmentationCoeff& coeff, Dtype* out) {
  const google::protobuf::Reflection* ref = coeff.GetReflection();
  const google::protobuf::Descriptor* desc = coeff.GetDescriptor();
  for (int fn = 0 ; fn < desc->field_count(); fn++) {
    const google::protobuf::FieldDescriptor* field_desc = desc->field(fn);
    float field_val = ref->GetFloat(coeff, field_desc);
    if (fabs(field_desc->default_value_float()) < 1e-3)
      out[fn] = field_val;
    else
      out[fn] = log(field_val);
//     LOG(INFO) << "writing field " << fn << ": " << out[fn];
  }
//   LOG(INFO) << "Finished writing params";
} 

template <typename Dtype>
void AugmentationLayerBase<Dtype>::array_to_coeff(const Dtype* in, AugmentationCoeff& coeff) {
  const google::protobuf::Reflection* ref = coeff.GetReflection();
  const google::protobuf::Descriptor* desc = coeff.GetDescriptor();
  for (int fn = 0 ; fn < desc->field_count(); fn++) {
    const google::protobuf::FieldDescriptor* field_desc = desc->field(fn);
    if (fabs(field_desc->default_value_float()) < 1e-3)
      ref->SetFloat(&coeff, field_desc, in[fn]);
    else
      ref->SetFloat(&coeff, field_desc, exp(in[fn]));
//     LOG(INFO) << "reading field " << fn << ": " << ref->GetFloat(coeff, field_desc);
  }
}


INSTANTIATE_CLASS(AugmentationLayerBase);
// template class AugmentationLayerBase<float>;
// template class AugmentationLayerBase<double>;

}  // namespace caffe

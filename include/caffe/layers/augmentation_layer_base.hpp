#ifndef AUGMENTATION_LAYER_BASE_HPP_
#define AUGMENTATION_LAYER_BASE_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Abstract Data Augmentation Layer
 *
 */

template <typename Dtype>
class AugmentationLayerBase {
public:
    class tTransMat {
        //tTransMat for Augmentation
        // | 0 2 4 |
        // | 1 3 5 |
    public:
        float t0, t2, t4;
        float t1, t3, t5;

        void toIdentity();

        void leftMultiply(float u0, float u1, float u2, float u3, float u4, float u5);

        void fromCoeff(AugmentationCoeff* coeff,int width,int height,int bottomwidth,int bottomheight);
        
        tTransMat inverse();
    };    

    class tChromaticCoeffs
    {
    public:
        float gamma;
        float brightness;
        float contrast;
        float color[3];

        void fromCoeff(AugmentationCoeff* coeff) { gamma=coeff->gamma(); brightness=coeff->brightness(); contrast=coeff->contrast(); color[0]=coeff->color1(); color[1]=coeff->color2(); color[2]=coeff->color3();  }

        bool needsComputation() { return gamma!=1 || brightness!=0 || contrast!=1 || color[0]!=1 || color[1]!=1 || color[2]!=1; }
    };

    class tChromaticEigenCoeffs
    {
    public:
        float pow_nomean0;
        float pow_nomean1;
        float pow_nomean2;
        float add_nomean0;
        float add_nomean1;
        float add_nomean2;
        float mult_nomean0;
        float mult_nomean1;
        float mult_nomean2;
        float pow_withmean0;
        float pow_withmean1;
        float pow_withmean2;
        float add_withmean0;
        float add_withmean1;
        float add_withmean2;
        float mult_withmean0;
        float mult_withmean1;
        float mult_withmean2;
        float lmult_pow;
        float lmult_add;
        float lmult_mult;
        float col_angle;

        void fromCoeff(AugmentationCoeff* coeff) {
            pow_nomean0=coeff->pow_nomean0();       pow_nomean1=coeff->pow_nomean1();       pow_nomean2=coeff->pow_nomean2();
            add_nomean0=coeff->add_nomean0();       add_nomean1=coeff->add_nomean1();       add_nomean2=coeff->add_nomean2();
            mult_nomean0=coeff->mult_nomean0();     mult_nomean1=coeff->mult_nomean1();     mult_nomean2=coeff->mult_nomean2();
            pow_withmean0=coeff->pow_withmean0();   pow_withmean1=coeff->pow_withmean1();   pow_withmean2=coeff->pow_withmean2();
            add_withmean0=coeff->add_withmean0();   add_withmean1=coeff->add_withmean1();   add_withmean2=coeff->add_withmean2();
            mult_withmean0=coeff->mult_withmean0(); mult_withmean1=coeff->mult_withmean1(); mult_withmean2=coeff->mult_withmean2();
            lmult_pow=coeff->lmult_pow();           lmult_add=coeff->lmult_add();           lmult_mult=coeff->lmult_mult();
            col_angle=coeff->col_angle(); }

        bool needsComputation() {
            return pow_nomean0!=1    || pow_nomean1!=1    || pow_nomean2!=1
                || add_nomean0!=0    || add_nomean1!=0    || add_nomean2!=0
                || mult_nomean0!=1   || mult_nomean1!=1   || mult_nomean2!=1
                || pow_withmean0!=1  || pow_withmean1!=1  || pow_withmean2!=1
                || add_withmean0!=0  || add_withmean1!=0  || add_withmean2!=0
                || mult_withmean0!=1 || mult_withmean1!=1 || mult_withmean2!=1
                || lmult_pow!=1      || lmult_add!=0      || lmult_mult!=1
                || col_angle!=0; }
    };

    class tEffectCoeffs
    {
    public:
        float fog_amount;
        float fog_size;
        float motion_blur_angle;
        float motion_blur_size;
        float shadow_nx;
        float shadow_ny;
        float shadow_distance;
        float shadow_strength;
        float noise;

        void fromCoeff(AugmentationCoeff* coeff) { fog_amount=coeff->fog_amount(); fog_size=coeff->fog_size(); motion_blur_angle=coeff->motion_blur_angle(); motion_blur_size=coeff->motion_blur_size(); shadow_nx=cos(coeff->shadow_angle()); shadow_ny=sin(coeff->shadow_angle()); shadow_distance=coeff->shadow_distance(); shadow_strength=coeff->shadow_strength(); noise=coeff->noise();}
        bool needsComputation() { return (fog_amount!=0 && fog_size!=0) || motion_blur_size>0 || shadow_strength>0 || noise>0; }
    };

    class tChromaticEigenSpace
    {
    public:
        // Note: these need to be floats for the CUDA atomic operations to work
        float mean_eig [3];
        float mean_rgb[3];
        float max_abs_eig[3];
        float max_rgb[3];
        float min_rgb[3];
        float max_l;
        float eigvec [9];
    };

    void clear_spatial_coeffs(AugmentationCoeff& coeff);
    void generate_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff);
    void generate_valid_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff,
                                       int width, int height, int cropped_width, int cropped_height, int max_num_tries = 50); 

    void clear_chromatic_coeffs(AugmentationCoeff& coeff);
    void generate_chromatic_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff);

    void clear_chromatic_eigen_coeffs(AugmentationCoeff& coeff);
    void generate_chromatic_eigen_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff);

    void clear_effect_coeffs(AugmentationCoeff& coeff);
    void generate_effect_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff);

    void clear_all_coeffs(AugmentationCoeff& coeff);
    void clear_defaults(AugmentationCoeff& coeff);

    void coeff_to_array(const AugmentationCoeff& coeff, Dtype* out);
    void array_to_coeff(const Dtype* in, AugmentationCoeff& coeff);
    void add_coeff_to_array(const AugmentationCoeff& coeff_in, Dtype* out_params);
};


}  // namespace caffe

#endif  // AUGMENTATION_LAYER_BASE_HPP_

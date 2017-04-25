#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

template <typename Dtype, typename Randtype>
Randtype caffe_rng_generate(const RandomGeneratorParameter& param, Dtype discount_coeff, Dtype prob0_value) {
  float spread;
  if (param.apply_schedule())
    spread = param.spread() * discount_coeff;
  else
    spread = param.spread();
  const std::string rand_type =  param.rand_type();
  //std::cout << rand_type << " " << rand_type.compare("uniform") << " " << rand_type.compare("gaussian") << " " << rand_type.compare("bernoulli");
  Randtype rand;
  if (rand_type.compare("uniform") == 0) {
    float tmp;
    if (spread > 0.)
      caffe_rng_uniform(1, param.mean() - spread, param.mean() + spread, &tmp);
    else
      tmp = param.mean();
    if (param.exp())
      tmp = exp(tmp);
    rand = static_cast<Randtype>(tmp);
  }
  else if (rand_type.compare("gaussian") == 0) {
    float tmp;
    if (spread > 0.)
      caffe_rng_gaussian(1, param.mean(), spread, &tmp);
    else
      tmp = param.mean();
    if (param.exp())
      tmp = exp(tmp);
    rand = static_cast<Randtype>(tmp);
  }
  else if (rand_type.compare("bernoulli") == 0) {
    int tmp;
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp);
    else
      tmp = 0;
    rand = static_cast<Randtype>(tmp);
  }
  else if (rand_type.compare("uniform_bernoulli") == 0) {
    float tmp1;
    int tmp2;

    // Eddy:
    // modified: a probability value of 0 will always return the default of prob0_value

    tmp2=1;
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp2);
    else
        tmp2=0;

    if(!tmp2) {
      if (!isnan(prob0_value))
        return prob0_value;
      else
        tmp1 = 0;
    } else {
      if (spread > 0.)
        caffe_rng_uniform(1, param.mean() - spread, param.mean() + spread, &tmp1);
      else
        tmp1 = param.mean();
    }

    if (param.exp())
      tmp1 = exp(tmp1);

    rand = static_cast<Randtype>(tmp1);
  }
  else if (rand_type.compare("gaussian_bernoulli") == 0) {
    float tmp1;
    int tmp2;

    // Eddy:
    // modified: a probability value of 0 will always return the default of prob0_value

    tmp2=1;
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp2);
    else
      tmp2=0;

    if(!tmp2) {
      if (!isnan(prob0_value))
        return prob0_value;
      else
        tmp1 = 0;
    } else {
      if (spread > 0.)
        caffe_rng_gaussian(1, param.mean(), spread, &tmp1);
      else
        tmp1 = param.mean();
    }

    if (param.exp())
      tmp1 = exp(tmp1);

    rand = static_cast<Randtype>(tmp1);
  }
  else {
    LOG(ERROR) << "Unknown random type " << rand_type;
    rand = NAN;
  }
  
  if(param.discretize()) rand = round(rand);
  rand = param.multiplier() * rand;
  
  return rand;
}

template float caffe_rng_generate<float,float>(const RandomGeneratorParameter& param, float discount_coeff, float prob0_value);
template bool caffe_rng_generate<float,bool>(const RandomGeneratorParameter& param, float discount_coeff, float prob0_value);
template float caffe_rng_generate<double,float>(const RandomGeneratorParameter& param, double discount_coeff, double prob0_value);
template double caffe_rng_generate<double,double>(const RandomGeneratorParameter& param, double discount_coeff, double prob0_value);
template bool caffe_rng_generate<double,bool>(const RandomGeneratorParameter& param, double discount_coeff, double prob0_value);

}

#ifndef CAFFE_OUTPUT_CPP_HPP_
#define CAFFE_OUTPUT_CPP_HPP_

#include "caffe/common.hpp"

namespace caffe {

bool readFloFile(std::string filename,float*& data,int& xSize,int &ySize);
void writeFloFile(std::string filename,const float* data,int xSize,int ySize);

void writePPM(std::string filename,const float* data,int xSize,int ySize,bool flipRGB=true,float scale=1.0);
void writePPMNorm(std::string filename,const float* data,int xSize,int ySize,bool flipRGB=true);

void writePGM(std::string filename,const float* data,int xSize,int ySize,float scale=1.0);
void writePGMNorm(std::string filename,const float* data,int xSize,int ySize);

void writeFloatFile(std::string filename,const float* data,int xSize,int ySize,int zSize);
void readFloatFile(std::string filename,float*& data,int& xSize,int& ySize,int& zSize);

}  // namespace caffe

#endif  // CAFFE_OUTPUT_CPP_HPP_

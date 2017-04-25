#include <caffe/util/output.hpp>
#include <stdio.h>
#include <string>
#include <limits>
using namespace std;

#include <caffe/caffe.hpp>

namespace caffe
{

inline float valclamp(float val) {
  return (val<0)?0:((val>255)?255:val);
}

bool readFloFile(string filename, float*& data, int& xSize, int &ySize)
{
    FILE *stream = fopen(filename.c_str(), "rb");
    if (stream == 0)
    {
        LOG(ERROR) << "Could not open or find file " << filename;
        return false;
    }

    try {
      float help;
      if(0 == fread(&help,sizeof(float),1,stream)) throw 0;
      if(0 == fread(&xSize,sizeof(int),1,stream)) throw 0;
      if(0 == fread(&ySize,sizeof(int),1,stream)) throw 0;

      data=new float[xSize*ySize*2];

      for (int y = 0; y < ySize; y++)
          for (int x = 0; x < xSize; x++) {
              if(0 == fread(&data[y*xSize+x],sizeof(float),1,stream)) throw 0;
              if(0 == fread(&data[y*xSize+x+xSize*ySize],sizeof(float),1,stream)) throw 0;
          }
    } catch(int err) {
      fclose(stream);
      LOG(FATAL) << "File corrupted: " << filename;
    }
    fclose(stream);

    return true;
}

void writeFloFile(string filename, const float* data, int xSize, int ySize)
{
    FILE *stream = fopen(filename.c_str(), "wb");

    // write the header
    fprintf(stream,"PIEH");
    fwrite(&xSize,sizeof(int),1,stream);
    fwrite(&ySize,sizeof(int),1,stream);

    // write the data
    for (int y = 0; y < ySize; y++)
        for (int x = 0; x < xSize; x++) {
            float u = data[y*xSize+x];
            float v = data[y*xSize+x+ySize*xSize];
            fwrite(&u,sizeof(float),1,stream);
            fwrite(&v,sizeof(float),1,stream);
        }
    fclose(stream);
}

void writePPM(std::string filename, const float* data, int xSize, int ySize, bool flipRGB, float scale)
{
    FILE* outimage = fopen(filename.c_str(), "wb");
    fprintf(outimage, "P6 \n");
    fprintf(outimage, "%d %d \n255\n", xSize,ySize);
    int size=xSize*ySize;

    int offset0=flipRGB?(2*size):(0*size);
    int offset1=flipRGB?(1*size):(1*size);
    int offset2=flipRGB?(0*size):(2*size);

    for (int i=0; i<size; i++)
        {
            float value1 = valclamp(((*(data+offset0+i))*scale));
            unsigned char aHelp = (unsigned char)value1;
            fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
            
            float value2 = valclamp(((*(data+offset1+i))*scale));
            aHelp = (unsigned char)value2;
            fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
            
            float value3 = valclamp(((*(data+offset2+i))*scale));
            aHelp = (unsigned char)value3;
            fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
        }
    fclose(outimage);
}

void writePPMNorm(std::string filename, const float* data, int xSize, int ySize, bool flipRGB)
{
    int size=xSize*ySize;
    float min=std::numeric_limits<float>::max();
    float max=std::numeric_limits<float>::min();

    int offset0=flipRGB?(2*size):(0*size);
    int offset1=flipRGB?(1*size):(1*size);
    int offset2=flipRGB?(0*size):(2*size);

    for (int i=0; i<3*size; i++)
        {
            if(data[i]<min)
                min=data[i];
            if(data[i]>max)
                max=data[i];
        }

    FILE* outimage = fopen(filename.c_str(), "wb");
    fprintf(outimage, "P6\n");
    fprintf(outimage, "%d %d \n255\n", xSize,ySize);
    for (int i=0; i<size; i++)
        {
            unsigned char aHelp = (unsigned char)(255*(*(data+offset0+i)-min)/(max-min));
            fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
            aHelp = (unsigned char)(255*(*(data+offset1+i)-min)/(max-min));
            fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
            aHelp = (unsigned char)(255*(*(data+offset2+i)-min)/(max-min));
            fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
        }
    fclose(outimage);
}

void writePGM(std::string filename, const float* data, int xSize, int ySize, float scale)
{
    FILE* outimage = fopen(filename.c_str(), "wb");

    fprintf(outimage, "P5\n");
    fprintf(outimage, "%d %d \n255\n", xSize,ySize);

    int size=xSize*ySize;
    for (int i=0; i<size; i++)
        {
            unsigned char aHelp = (unsigned char)(*(data+i)*scale);
            fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
        }
    fclose(outimage);
}

void writePGMNorm(std::string filename, const float* data, int xSize, int ySize)
{
    int size=xSize*ySize;
    float min=std::numeric_limits<float>::max();
    float max=std::numeric_limits<float>::min();

    for (int i=0; i<size; i++)
        {
            if(data[i]<min)
                min=data[i];
            if(data[i]>max)
                max=data[i];
        }

    FILE* outimage = fopen(filename.c_str(), "wb");
    fprintf(outimage, "P5\n");
    fprintf(outimage, "%d %d \n255\n", xSize,ySize);
    for (int i=0; i<size; i++)
        {
            unsigned char aHelp = (unsigned char)(255*(*(data+i)-min)/(max-min));
            fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
        }
    fclose(outimage);
}

#define fail(text)  LOG(FATAL) << text;

void writeFloatFile(std::string filename, const float* data, int xSize, int ySize, int zSize)
{
    int size=xSize*ySize*zSize;

    FILE* f=fopen(filename.c_str(),"wb");
    if(!f)
        fail("Cannot open file for writing: "+filename);

    fprintf(f,"float\n");
    fprintf(f,"%d\n",3);
    fprintf(f,"%d\n",xSize);
    fprintf(f,"%d\n",ySize);
    fprintf(f,"%d\n",zSize);

    fwrite(data,size,sizeof(float),f);

    fclose(f);
}

void readFloatFile(std::string filename, float*& data, int& xSize, int& ySize, int& zSize)
{
    FILE* f=fopen(filename.c_str(),"rb");
    
    if(!f) fail("Cannot open file: "+filename);

    char buffer[128];
    try {
      if(NULL == fgets(buffer,128,f)) {throw 0;}; 
      if(strcmp(buffer,"float\n")!=0)
          fail("File has invalid format: "+filename);

      if(NULL == fgets(buffer,128,f)) {throw 0;}; 
      int dim=atoi(buffer);
      int dimX=1;
      int dimY=1;
      int dimZ=1;
      if(dim==3)
      {
        if(NULL == fgets(buffer,128,f)) {throw 0;}; dimX=atoi(buffer);
        if(NULL == fgets(buffer,128,f)) {throw 0;}; dimY=atoi(buffer);
        if(NULL == fgets(buffer,128,f)) {throw 0;}; dimZ=atoi(buffer);
      }
      else if(dim==2)
      {
        if(NULL == fgets(buffer,128,f)) {throw 0;}; dimX=atoi(buffer);
        if(NULL == fgets(buffer,128,f)) {throw 0;}; dimY=atoi(buffer);
      }
      else {
        fail("File not 3-dimensional: "+filename);
      }

      xSize=dimX;
      ySize=dimY;
      zSize=dimZ;
      int size=xSize*ySize*zSize;
      data=new float[size];
      if(0 == fread(data,size,sizeof(float),f)) throw 0;
    } catch(int err) {
      fclose(f);
      LOG(FATAL) << "File corrupted: " << filename;
    }
    
    fclose(f);
}

}

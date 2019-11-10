#pragma once

#include <include/jpegloader.h>
#include <include/timer.h>

class Labwork
{

private:
    JpegLoader jpegLoader;
    JpegInfo *inputImage;
    JpegInfo *inputImage1;
    char *outputImage;

public:
    void loadInputImage(std::string inputFileName);
    JpegInfo *loadImage(std::string inputFileName1);
    void saveOutputImage(std::string outputFileName);

    void labwork1_CPU();
    void labwork1_OpenMP();

    void labwork2_GPU();

    void labwork3_GPU();

    void labwork4_GPU();

    void labwork5_CPU();
    void labwork5_GPU();
    void labwork5_GPU_shared_memmory();

    void labwork6a_GPU(int thres);
    void labwork6b_GPU(int brightness);
    void labwork6c_GPU(float blendRatio, JpegInfo *inputImage1);

    void labwork7_GPU();

    void labwork8_GPU();

    void labwork9_GPU();

    void labwork10_GPU();
};
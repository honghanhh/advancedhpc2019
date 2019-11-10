#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <time.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv)
{
    printf("USTH ICT Master 2019, Advanced Programming for HPC.\n");
    if (argc < 2)
    {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // Pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2)
    {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    //Initialize extra parameters for labwork 6
    int option = atoi(argv[3]);

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum)
    {
    case 1:
        labwork.labwork1_CPU();
        labwork.saveOutputImage("labwork2-cpu-out.jpg");
        printf("Labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        timer.start();
        labwork.labwork1_OpenMP();
        labwork.saveOutputImage("labwork2-openmp-out.jpg");
        break;
    case 2:
        labwork.labwork2_GPU();
        break;
    case 3:
        labwork.labwork3_GPU();
        labwork.saveOutputImage("labwork3-gpu-out.jpg");
        break;
    case 4:
        timer.start();
        labwork.labwork4_GPU();
        printf("Labwork 4 ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork4-gpu-out.jpg");
        break;
    case 5:

        float timeCPU, timeGPUShare, timeGPUNonShare;
        timer.start();
        labwork.labwork5_CPU();
        timeCPU = timer.getElapsedTimeInMilliSec();

        labwork.saveOutputImage("labwork5-cpu-out.jpg");

        timer.start();
        labwork.labwork5_GPU();
        timeGPUNonShare = timer.getElapsedTimeInMilliSec();

        labwork.saveOutputImage("labwork5-gpu-out.jpg");

        timer.start();
        labwork.labwork5_GPU_shared_memmory();
        timeGPUShare = timer.getElapsedTimeInMilliSec();

        labwork.saveOutputImage("labwork5-gpu-shared_mem_out.jpg");

        printf("Labwork 5 CPU ellapsed %.1fms\n", lwNum, timeCPU);
        printf("Labwork 5 GPU with shared memory ellapsed %.1fms\n", lwNum, timeGPUShare);
        printf("Labwork 5 GPU without shared memory ellapsed %.1fms\n", lwNum, timeGPUNonShare);
        printf("GPU without shared memory is faster than CPU: %.2f times\n", timeCPU / timeGPUNonShare);
        printf("GPU with shared memory is faster than CPU: %.2f times\n", timeCPU / timeGPUShare);
        printf("GPU with shared memory is faster than GPU without shared memory: %.2f times\n", timeGPUNonShare / timeGPUShare);
        break;
    case 6:

        float timeBinarize, timeBright, timeBlend;
        int parama, paramb;
        float paramc;

        switch (option)
        {
        case 0:
            parama = atoi(argv[4]);
            timer.start();
            labwork.labwork6a_GPU(parama);
            timeBinarize = timer.getElapsedTimeInMilliSec();

            labwork.saveOutputImage("labwork6a-gpu-out.jpg");
            printf("Labwork 6a (Binarization) ellapsed %.1fms\n", lwNum, timeBinarize);
            break;
        case 1:
            paramb = atoi(argv[4]);
            timer.start();
            labwork.labwork6b_GPU(paramb);
            timeBright = timer.getElapsedTimeInMilliSec();

            printf("Labwork 6b (Blending) ellapsed %.1fms\n", lwNum, timeBright);
            labwork.saveOutputImage("labwork6b-gpu-out.jpg");
            break;
        case 2:
            paramc = atof(argv[4]);
            std::string inputFilename1;
            JpegInfo *inputImage1;

            inputFilename1 = std::string(argv[5]);
            inputImage1 = labwork.loadImage(inputFilename1);

            timer.start();
            labwork.labwork6c_GPU(paramc, inputImage1);
            timeBlend = timer.getElapsedTimeInMilliSec();

            printf("Labwork 6c (GrayScaling) ellapsed %.1fms\n", lwNum, timeBlend);
            labwork.saveOutputImage("labwork6c-gpu-out.jpg");
            break;
        }
        break;
    case 7:
        labwork.labwork7_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork7-gpu-out.jpg");
        break;
    case 8:
        labwork.labwork8_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork8-gpu-out.jpg");
        break;
    case 9:
        labwork.labwork9_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork9-gpu-out.jpg");
        break;
    case 10:
        labwork.labwork10_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork10-gpu-out.jpg");
        break;
    }
    printf("Labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName)
{
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName)
{
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

JpegInfo *Labwork::loadImage(std::string fileName)
{
    return jpegLoader.load(fileName);
}

void Labwork::labwork1_CPU()
{
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++)
    { // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++)
        {
            outputImage[i * 3] = (char)(((int)inputImage->buffer[i * 3] + (int)inputImage->buffer[i * 3 + 1] +
                                         (int)inputImage->buffer[i * 3 + 2]) /
                                        3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP()
{
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
// do something here
#pragma omp master
    for (int j = 0; j < 100; j++)
    { // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++)
        {
            outputImage[i * 3] = (char)(((int)inputImage->buffer[i * 3] + (int)inputImage->buffer[i * 3 + 1] +
                                         (int)inputImage->buffer[i * 3 + 2]) /
                                        3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major)
    {
    case 2: // Fermi
        if (devProp.minor == 1)
            cores = mp * 48;
        else
            cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if (devProp.minor == 1)
            cores = mp * 128;
        else if (devProp.minor == 0)
            cores = mp * 64;
        else
            printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

void Labwork::labwork2_GPU()
{
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
        // Device name
        printf("GPU #%d\n", i);
        printf("GPU name: %s\n", prop.name);
        //Core info
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Number of cores: %d\n", getSPcores(prop));
        printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Warp Size: %d\n", prop.warpSize);
        //Memory info
        printf("Memory Clock Rate: %d\n", prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\nDevices", prop.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

__global__ void grayscale(uchar3 *input, uchar3 *output)
{
    // this will execute in a device core
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU()
{
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    // Processing
    int blockSize = 512;
    int numBlock = pixelCount / blockSize;
    grayscale<<<numBlock, blockSize>>>(devInput, devOutput);
    // Copy CUDA Memory from GPU to CPU
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void grayscale2D(uchar3 *input, uchar3 *output, int width, int height)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;

    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;

    int tid = tidY * width + tidX;

    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU()
{
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    // Processing
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);
    grayscale2D<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);
    // Copy CUDA Memory from GPU to CPU
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

void Labwork::labwork5_CPU()
{
    int kernel[] = {0, 0, 1, 2, 1, 0, 0,
                    0, 3, 13, 22, 13, 3, 0,
                    1, 13, 59, 97, 59, 13, 1,
                    2, 22, 97, 159, 97, 22, 2,
                    1, 13, 59, 97, 59, 13, 1,
                    0, 3, 13, 22, 13, 3, 0,
                    0, 0, 1, 2, 1, 0, 0};

    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));
    for (int rows = 0; rows < inputImage->height; rows++)
    {
        for (int columns = 0; columns < inputImage->width; columns++)
        {
            int sum = 0; // Normalization
            int c = 0;   // Constant
            for (int y = -3; y <= 3; y++)
            {
                for (int x = -3; x <= 3; x++)
                {
                    int i = columns + x;
                    int j = rows + y;
                    if (i < 0 || i >= inputImage->width || j < 0 || j >= inputImage->height)
                        continue;
                    int tid = i + j * inputImage->width;
                    char gray = (char)(((int)inputImage->buffer[tid * 3] + (int)inputImage->buffer[tid * 3 + 1] +
                                        (int)inputImage->buffer[tid * 3 + 2]) /
                                       3);
                    int coefficient = kernel[(y + 3) * 7 + x + 3];
                    sum += gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int positionOut = rows * inputImage->width + columns;
            if (positionOut < pixelCount)
            {
                outputImage[positionOut * 3] = outputImage[positionOut * 3 + 1] = outputImage[positionOut * 3 + 2] = sum;
            }
        }
    }
}

// Blur kernel for non-shared memory
__global__ void blurGaussianConvNonShared(uchar3 *input, uchar3 *output, int width, int height)
{

    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;
    int tid = tidX + tidY * width;

    int kernel[] = {0, 0, 1, 2, 1, 0, 0,
                    0, 3, 13, 22, 13, 3, 0,
                    1, 13, 59, 97, 59, 13, 1,
                    2, 22, 97, 159, 97, 22, 2,
                    1, 13, 59, 97, 59, 13, 1,
                    0, 3, 13, 22, 13, 3, 0,
                    0, 0, 1, 2, 1, 0, 0};

    int sum = 0; // Normalization
    int c = 0;   // Constant
    for (int y = -3; y < 3; y++)
    {
        for (int x = -3; x < 3; x++)
        {
            int i = tidX + x;
            int j = tidY + y;
            if (i < 0 || i >= width || j < 0 || j >= height)
                continue;
            int tid = i + j * width;
            unsigned char gray = (input[tid].x + input[tid].y + input[tid].z) / 3;
            int coefficient = kernel[(y + 3) * 7 + x + 3];
            sum += gray * coefficient;
            c += coefficient;
        }
    }

    sum /= c;
    output[tid].z = output[tid].y = output[tid].x = sum;
}

void Labwork::labwork5_GPU()
{

    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));

    // Allocate memory for the output on the host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));

    // Copy InputImage from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    blurGaussianConvNonShared<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

// Blur kernel for shared memory
__global__ void blurGaussianConvNonShared(uchar3 *input, uchar3 *output, int *kernel, int width, int height)
{

    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;
    int tid = tidX + tidY * width;

    __shared__ int sharedKernel[49];
    int localtid = threadIdx.x + threadIdx.y * blockDim.x;
    if (localtid < 49)
    {
        sharedKernel[localtid] = kernel[localtid];
    }
    __syncthreads();

    int sum = 0; // Normalization
    int c = 0;   // Constant
    for (int y = -3; y < 3; y++)
    {
        for (int x = -3; x < 3; x++)
        {
            int i = tidX + x;
            int j = tidY + y;
            if (i < 0 || i >= width || j < 0 || j >= height)
                continue;
            int tid = i + j * width;
            unsigned char gray = (input[tid].x + input[tid].y + input[tid].z) / 3;
            int coefficient = sharedKernel[(y + 3) * 7 + x + 3];
            sum += gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;
    output[tid].z = output[tid].y = output[tid].x = sum;
}

void Labwork::labwork5_GPU_shared_memmory()
{

    int kernel[] = {0, 0, 1, 2, 1, 0, 0,
                    0, 3, 13, 22, 13, 3, 0,
                    1, 13, 59, 97, 59, 13, 1,
                    2, 22, 97, 159, 97, 22, 2,
                    1, 13, 59, 97, 59, 13, 1,
                    0, 3, 13, 22, 13, 3, 0,
                    0, 0, 1, 2, 1, 0, 0};
    int *shareKernel;

    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
    cudaMalloc(&shareKernel, sizeof(kernel));

    // Allocate memory for the output on the host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));

    // Copy InputImage from CPU (host) to GPU (device)
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Copy Kernel into shared memory
    cudaMemcpy(shareKernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice);

    // Processing
    blurGaussianConvNonShared<<<gridSize, blockSize>>>(devInput, devOutput, shareKernel, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(shareKernel);
}

__global__ void binarization(uchar3 *input, uchar3 *output, int width, int height, int thres)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;

    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;

    int tid = tidY * width + tidX;

    //Define a threshold
    unsigned char binary = (input[tid].x + input[tid].y + input[tid].z) / 3;
    binary = min(binary / thres, 1) * 255;

    output[tid].z = output[tid].y = output[tid].x = binary;
}

void Labwork::labwork6a_GPU(int thres)
{
    // int thres = 128;

    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));

    // Allocate memory for the output on the host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));

    // Copy InputImage from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    binarization<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, thres);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void brightnessControl(uchar3 *input, uchar3 *output, int width, int height, int brightness)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;

    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;

    int tid = tidY * width + tidX;

    //Define a brightness
    unsigned char red = min(max(input[tid].x + brightness, 0), 255);
    unsigned char green = min(max(input[tid].y + brightness, 0), 255);
    unsigned char blue = min(max(input[tid].z + brightness, 0), 255);

    output[tid].x = red;
    output[tid].y = green;
    output[tid].z = blue;
}

void Labwork::labwork6b_GPU(int brightness)
{
    // int brightness = 100;

    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));

    // Allocate memory for the output on the host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));

    // Copy InputImage from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    brightnessControl<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, brightness);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void blendImages(uchar3 *input, uchar3 *input1, uchar3 *output, int width, int height, float blendRatio)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;

    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;

    int tid = tidY * width + tidX;

    //Define a blending
    unsigned char red = input[tid].x * blendRatio + input1[tid].x * (1 - blendRatio);
    unsigned char green = input[tid].y * blendRatio + input1[tid].y * (1 - blendRatio);
    unsigned char blue = input[tid].z * blendRatio + input1[tid].z * (1 - blendRatio);

    output[tid].x = red;
    output[tid].y = green;
    output[tid].z = blue;
}

void Labwork::labwork6c_GPU(float blendRatio, JpegInfo *inputImage1)
{

    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devInput1;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devInput1, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));

    // Allocate memory for the output on the host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));

    // Copy InputImage from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(devInput1, inputImage1->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    blendImages<<<gridSize, blockSize>>>(devInput, devInput1, devOutput, inputImage->width, inputImage->height, blendRatio);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

void Labwork::labwork7_GPU()
{
}

void Labwork::labwork8_GPU()
{
}

void Labwork::labwork9_GPU()
{
}

void Labwork::labwork10_GPU()
{
}

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
        //Initialize extra parameters for labwork 6
        int option;
        option = atoi(argv[3]);
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

            printf("Labwork 6b (Brightness) ellapsed %.1fms\n", lwNum, timeBright);
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

            printf("Labwork 6c (Blending) ellapsed %.1fms\n", lwNum, timeBlend);
            labwork.saveOutputImage("labwork6c-gpu-out.jpg");
            break;
        }
        break;
    case 7:
        timer.start();
        labwork.labwork7_GPU();
        printf("[ALGO ONLY] Labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork7-gpu-out.jpg");
        break;
    case 8:
        timer.start();
        labwork.labwork8_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork8-gpu-out.jpg");
        break;
    case 9:
        timer.start();
        labwork.labwork8_GPU();
        printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
        labwork.saveOutputImage("labwork9-gpu-out.jpg");
        break;
    case 10:
        timer.start();
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

__global__ void greyscale1(unsigned char *input, unsigned char *output, int width, int height)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;
    int tid = tidY * width + tidX;

    output[tid] = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2]) / 3;
}

__global__ void maxIntensity(unsigned char *input, unsigned char *output, int count)
{
    // Dynamic shared memory size, allocated in host
    extern __shared__ unsigned char cache[];

    // Cache the block content
    int blockSize = blockDim.x * blockDim.y;
    int localId = threadIdx.x + blockDim.x * threadIdx.y;
    int tid = blockIdx.x * blockSize + localId;

    if (tid < count)
    {
        cache[localId] = input[tid];
    }
    else
    {
        cache[localId] = 0;
    }

    __syncthreads();

    // Reduction in cache
    for (int s = 1; s < blockSize; s *= 2)
    {
        if (localId % (s * 2) == 0)
        {
            cache[localId] = max(cache[localId], cache[localId + s]);
        }

        __syncthreads();
    }
    // Only first thread writes back
    if (localId == 0)
    {
        output[blockIdx.x] = cache[0];
    }
}

__global__ void minIntensity(unsigned char *input, unsigned char *output, int count)
{
    // Dynamic shared memory size, allocated in host
    extern __shared__ unsigned char cache[];

    // Cache the block content
    int blockSize = blockDim.x * blockDim.y;
    int localId = threadIdx.x + blockDim.x * threadIdx.y;
    int tid = blockIdx.x * blockSize + localId;

    if (tid < count)
    {
        cache[localId] = input[tid];
    }
    else
    {
        cache[localId] = 255;
    }

    __syncthreads();

    // Reduction in cache
    for (int s = 1; s < blockSize; s *= 2)
    {
        if (localId % (s * 2) == 0)
        {
            cache[localId] = min(cache[localId], cache[localId + s]);
        }

        __syncthreads();
    }

    // Only first thread writes back
    if (localId == 0)
    {
        output[blockIdx.x] = cache[0];
    }
}

__global__ void grayscaleStretch(unsigned char *input, char *output, unsigned char *max, unsigned char *min, int width, int height)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;
    int tid = tidY * width + tidX;

    unsigned char greyStretched = ((float)(input[tid] - min[0]) / (max[0] - min[0])) * 255;

    output[tid * 3] = output[tid * 3 + 1] = output[tid * 3 + 2] = greyStretched;
}

void Labwork::labwork7_GPU()
{
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    unsigned char *devInput, *devGrey;
    char *devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devGrey, pixelCount);
    cudaMalloc(&devOutput, pixelCount * 3);

    // Allocate memory for the output on the host
    outputImage = (char *)malloc(pixelCount * 3);

    // Copy InputImage from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    greyscale1<<<gridSize, blockSize>>>(devInput, devGrey, inputImage->width, inputImage->height);

    // Start reduction
    int blockThreads = blockSize.x * blockSize.y;
    int reduce = (pixelCount + blockThreads - 1) / blockThreads;
    int swap = 0;

    unsigned char *devMax0, *devMax1, *devMin0, *devMin1;
    unsigned char *maxArrPointer[2], *minArrPointer[2];

    cudaMalloc(&devMax0, reduce);
    cudaMalloc(&devMin0, reduce);
    maxArrPointer[swap] = devMax0;
    minArrPointer[swap] = devMin0;

    maxIntensity<<<reduce, blockSize, blockThreads>>>(devGrey, devMax0, pixelCount);
    minIntensity<<<reduce, blockSize, blockThreads>>>(devGrey, devMin0, pixelCount);

    int temp = reduce;                                   //Count
    reduce = (reduce + blockThreads - 1) / blockThreads; //Reduce grid size

    cudaMalloc(&devMax1, reduce);
    cudaMalloc(&devMin1, reduce);
    maxArrPointer[!swap] = devMax1;
    minArrPointer[!swap] = devMin1;

    while (reduce > 1)
    {
        maxIntensity<<<reduce, blockSize, blockThreads>>>(maxArrPointer[swap], maxArrPointer[!swap], temp);
        minIntensity<<<reduce, blockSize, blockThreads>>>(minArrPointer[swap], minArrPointer[!swap], temp);

        temp = reduce;
        reduce = (reduce + blockThreads - 1) / blockThreads;
        swap = !swap;
    }

    maxIntensity<<<1, blockSize, blockThreads>>>(maxArrPointer[swap], maxArrPointer[!swap], temp);
    minIntensity<<<1, blockSize, blockThreads>>>(minArrPointer[swap], minArrPointer[!swap], temp);
    // end

    grayscaleStretch<<<gridSize, blockSize>>>(devGrey, devOutput, maxArrPointer[!swap], minArrPointer[!swap], inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    //Cleaning
    cudaFree(devInput);
    cudaFree(devGrey);
    cudaFree(devOutput);
    cudaFree(devMax0);
    cudaFree(devMax1);
    cudaFree(devMin0);
    cudaFree(devMin1);
}

__global__ void rgv2hsv(uchar3 *input, int *H, float *S, float *V, int width, int height)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;
    int tid = tidY * width + tidX;

    // Preparation

    // Scaling
    float R = input[tid].x / 255.0f;
    float G = input[tid].y / 255.0f;
    float B = input[tid].z / 255.0f;

    // Find max and min
    float max_rg = max(R, G);
    float min_rg = min(R, G);
    float maxV = max(max_rg, B);
    float minV = min(min_rg, B);
    float delta = maxV - minV;

    // Conversion
    V[tid] = maxV;

    if (delta == 0.0f)
    {
        H[tid] = 0;
        S[tid] = 0.0f; // Saturation when delta = 0
    }
    else
    {
        // Saturation conversion
        S[tid] = delta / maxV;

        // Hue conversion
        if (maxV == R)
        {
            H[tid] = 60.0f * fmod((G - B) / delta, 6.0f);
        }
        else if (maxV == G)
        {
            H[tid] = 60.0f * ((B - R) / delta + 2.0f);
        }
        else
        {
            H[tid] = 60.0f * ((R - G) / delta + 4.0f);
        }

        if (H[tid] < 0)
        {
            H[tid] = 360 + H[tid];
        }
    }
}

__global__ void hsv2rgb(int *H, float *S, float *V, uchar3 *output, int width, int height)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;
    int tid = tidY * width + tidX;

    float h = H[tid];
    float s = S[tid];
    float v = V[tid];

    // Preparation
    float d = h / 60;
    float hi = (int)fmodf(d, 6.0);
    float f = d - hi;
    float l = v * (1 - s);
    float m = v * (1 - (f * s));
    float n = v * (1 - ((1 - f) * s));

    // Conversion
    float R, G, B;
    if (h >= 0 and h < 60)
    {
        R = v;
        G = n;
        B = l;
    }
    else if (h >= 60 and h < 120)
    {
        R = m;
        G = v;
        B = l;
    }
    else if (h >= 120 and h < 180)
    {
        R = l;
        G = v;
        B = n;
    }
    else if (h >= 180 and h < 240)
    {
        R = l;
        G = m;
        B = v;
    }
    else if (h >= 240 and h < 300)
    {
        R = n;
        G = l;
        B = v;
    }
    else if (h >= 300 and h < 360)
    {
        R = v;
        G = l;
        B = m;
    }
    else
    {
        R = 1;
        G = 1;
        B = 1;
    }

    output[tid].x = R * 255;
    output[tid].y = G * 255;
    output[tid].z = B * 255;
}

void Labwork::labwork8_GPU()
{
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    dim3 blockSize = dim3(16, 16);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    uchar3 *devInput;
    uchar3 *devOutput;
    // Define Hue, Saturation, Value
    int *devH;
    float *devS, *devV;
    // Allocate CUDA memory
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devH, pixelCount * sizeof(int));
    cudaMalloc(&devS, pixelCount * sizeof(float));
    cudaMalloc(&devV, pixelCount * sizeof(float));

    // Allocate memory for the output on the host
    outputImage = (char *)malloc(pixelCount * sizeof(uchar3));

    // Copy InputImage from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    rgv2hsv<<<gridSize, blockSize>>>(devInput, devH, devS, devV, inputImage->width, inputImage->height);
    hsv2rgb<<<gridSize, blockSize>>>(devH, devS, devV, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(devH);
    cudaFree(devS);
    cudaFree(devV);
}

__global__ void histogramGather(uchar3 *input, unsigned int **output, int width, int height)
{
    unsigned int histoL[256] = {0};
    for (int i = 0; i < height; i++)
    {
        int j = input[blockIdx.x * height + i].x;
        histoL[j]++;
    }
    for (int i = 0; i < 256; i++)
    {
        output[blockIdx.x][i] = histoL[i];
    }
}

__global__ void histogramReduction(unsigned int **input, int *output, int width, int height)
{
    // Dynamic shared memory size, allocated in host
    __shared__ unsigned int cache[256];

    // Cache the block content
    unsigned int localtid = threadIdx.x;
    cache[localtid] = 0;

    __syncthreads();

    // Reduction in cache
    for (int i = 0; i < width; i++)
    {
        cache[localtid] += input[i][localtid];
    }
    __syncthreads();

    // Only first thread writes back
    if (localtid == 0)
    {
        for (int i = 0; i < 256; i++)
        {
            output[i] = cache[i];
        }
    }
}

__global__ void cdfCalculation(int *h, int pixelCount)
{
    int cdfMin = 0;
    int cdfCumul = 0;
    for (int i = 0; i < 256; i++)
    {
        if (cdfMin == 0)
        {
            cdfMin = h[i];
        }
        cdfCumul += h[i];
        h[i] = round((double)(cdfCumul - cdfMin) / (pixelCount - cdfMin) * 255.0);
    }
}

__global__ void equalization(uchar3 *input, int *h, uchar3 *output, int width, int height)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;
    int tid = tidX + tidY * width;

    unsigned char g = h[input[tid].x];
    output[tid].x = output[tid].y = output[tid].z = g;
}

void Labwork::labwork9_GPU()
{
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    uchar3 *devInput;
    uchar3 *devOutput;
    uchar3 *tempOutput;
    unsigned int *hist2Local[inputImage->width];
    int *hist2Final;
    int *hostHisto = (int *)calloc(256, sizeof(int));

    // Allocate CUDA memory
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
    cudaMalloc(&tempOutput, pixelCount * sizeof(uchar3));
    // cudaMalloc(&hist2Local, inputImage->width * 256);
    cudaMalloc(&hist2Final, 256 * sizeof(int));
    for (int i = 0; i < inputImage->width; i++)
    {
        cudaMalloc(&hist2Local[i], 256 * sizeof(unsigned int));
    }

    // Allocate memory for the output on the host
    outputImage = (char *)malloc(pixelCount * sizeof(uchar3));

    // Copy InputImage from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    grayscale2D<<<gridSize, blockSize>>>(devInput, tempOutput, inputImage->width, inputImage->height);
    histogramGather<<<inputImage->width, 1>>>(tempOutput, hist2Local, inputImage->width, inputImage->height);
    histogramReduction<<<1, 256>>>(hist2Local, hist2Final, inputImage->width, inputImage->height);
    cdfCalculation<<<1, 1>>>(hist2Final, pixelCount);
    equalization<<<gridSize, blockSize>>>(tempOutput, hist2Final, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(tempOutput);
    cudaFree(hist2Local);
    cudaFree(hist2Final);
}

__global__ void kuwahara(uchar3 *input, uchar3 *output, int width, int height, int winSize)
{
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidX >= width)
        return;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidY >= height)
        return;
    int tid = tidY * width + tidX;

    double window[4] = {0.0};
    double SD[4] = {0.0};
    int meanRGB[4][3] = {0};
    int pxCount[4] = {0};
    int winPos;

    for (int x = 1 - winSize; x <= winSize - 1; x++)
    {
        for (int y = 1 - winSize; y <= winSize - 1; y++)
        {
            int rows = tidX + x;
            int columns = tidY + y;
            if (rows < 0 || rows >= width || columns < 0 || columns >= height)
                continue;
            int positionOut = rows + columns * width;

            int red = input[positionOut].x;
            int green = input[positionOut].y;
            int blue = input[positionOut].z;

            if (x >= 0 && y <= 0)
            {
                winPos = 3; // bottom right
            }

            if (x <= 0 && y <= 0)
            {
                winPos = 2; // bottom left
            }

            if (x >= 0 && y >= 0)
            {
                winPos = 1; //top right
            }

            if (x <= 0 && y >= 0)
            {
                winPos = 0; // top left
            }
            meanRGB[winPos][0] += red;
            meanRGB[winPos][1] += green;
            meanRGB[winPos][2] += blue;

            window[winPos] += max(red, max(green, blue));
            pxCount[winPos]++;

            SD[winPos] += pow((max(red, max(green, blue)) - window[winPos]), 2.0);
        }
    }

    for (int i = 0; i < 4; i++)
    {
        SD[i] = sqrt(SD[i] / (pxCount[i]));
        window[i] /= pxCount[i];
        for (int j = 0; j < 3; j++)
        {
            meanRGB[i][j] /= pxCount[i];
        }
    }

    double minSD = min(SD[0], min(SD[1], min(SD[2], SD[3])));
    if (minSD == SD[0])
        tidX = 0;
    else if (minSD == SD[1])
        tidX = 1;
    else if (minSD == SD[2])
        tidX = 2;
    else
        tidX = 3;

    output[tid].x = meanRGB[tidX][0];
    output[tid].y = meanRGB[tidX][1];
    output[tid].z = meanRGB[tidX][2];
}

void Labwork::labwork10_GPU()
{
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    int winSize = 32;

    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x - 1) / blockSize.x, (inputImage->height + blockSize.y - 1) / blockSize.y);

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));

    // Allocate memory for the output on the host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));

    // Copy InputImage from CPU (host) to GPU (device)
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    kuwahara<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, winSize);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

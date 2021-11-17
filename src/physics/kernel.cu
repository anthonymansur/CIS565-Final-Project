#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include <cuda.h>
#include "../utilityCore.hpp"
#include "kernel.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

void checkCUDAError(const char* msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*****************
* Configuration *
*****************/

#define blockSize 128
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

glm::vec3* dev_pos;

/***************
* Copy to VBO *
****************/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale = -1.0f / s_scale;

    if (index < N) {
        vbo[4 * index + 0] = pos[index].x * c_scale;
        vbo[4 * index + 1] = pos[index].y * c_scale;
        vbo[4 * index + 2] = pos[index].z * c_scale;
        vbo[4 * index + 3] = 1.0f;
    }
}

void Simulation::copyToVBO(float* vbodptr_positions) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
    checkCUDAErrorWithLine("copyToVBO failed!");

    cudaDeviceSynchronize();
}


/******************
* initSimulation *
******************/

//__host__ __device__ unsigned int hash(unsigned int a) {
//    a = (a + 0x7ed55d16) + (a << 12);
//    a = (a ^ 0xc761c23c) ^ (a >> 19);
//    a = (a + 0x165667b1) + (a << 5);
//    a = (a + 0xd3a2646c) ^ (a << 9);
//    a = (a + 0xfd7046c5) + (a << 3);
//    a = (a ^ 0xb55a4f09) ^ (a >> 16);
//    return a;
//}
//
//__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
//    thrust::default_random_engine rng(hash((int)(index * time)));
//    thrust::uniform_real_distribution<float> unitDistrib(-1, 1);
//
//    return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
//}
//
//__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3* arr, float scale) {
//    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//    if (index < N) {
//        glm::vec3 rand = generateRandomVec3(time, index);
//        arr[index].x = scale * rand.x;
//        arr[index].y = scale * rand.y;
//        arr[index].z = scale * rand.z;
//    }
//}

__global__ void kernGeneratePlanePosArray(int time, int N, glm::vec3* arr, float scale) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < N) {
        arr[index].x = 0.0f;
        arr[index].y = 1.0f;
        arr[index].z = 0.0f;

        arr[index + 1].x = 10.0f;
        arr[index + 1].y = 1.0f;
        arr[index + 1].z = 0.0f;

        arr[index + 2].x = 10.0f;
        arr[index + 2].y = 1.0f;
        arr[index + 2].z = 10.0f;

        arr[index + 3].x = 0.0f;
        arr[index + 3].y = 1.0f;
        arr[index + 3].z = 10.0f;
    }
}



void Simulation::initSimulation(int N)
{
    numObjects = N;
    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_pos failed");

    kernGeneratePlanePosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale);
    checkCUDAErrorWithLine("kernGenerateRandomPosArray failed");

    cudaDeviceSynchronize();
}

/******************
* stepSimulation *
******************/
void Simulation::stepSimulation(float dt)
{
    // TODO: implement
}

/******************
* endSimulation *
******************/
void Simulation::endSimulation()
{
    cudaFree(dev_pos);
}
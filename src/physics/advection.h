#ifndef ADVECTION_H
#define ADVECTION_H

#include <cstdio>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../errors.h"
#include "vec3.h"
#include "pressure.h"
#include "kernel.h"

struct uchar4;
// struct BC that contains all the boundary conditions
typedef struct {
    int x, y; // x and y location of pipe center
    float rad; // radius of pipe
    int chamfer; // chamfer
    float t_s, t_a, t_g; // temperatures in pipe, air, ground
} BC;

int blocksNeeded(int N_i, int M_i);
__device__ int flatten(int3 gridCount, int col, int row, int z);
__device__ int vflatten(int3 gridCount, int col, int row, int z);
__device__ int flatten(int col, int row, int z, int width, int height, int depth);
__device__ unsigned char clip(int n);

__device__ float3 operator+(const float3& a, const float3& b);
__device__ float3 operator-(const float3& a, const float3& b);
__device__ float3 operator*(const float3& a, const float& b);
__device__ float3 operator*(const float& b, const float3& a);

/**
* @brief Computes contribution to velocity (wind speed) update due to turbulent forces
* 
* @param gridCount of simulation space in x, y, z directions
* @param blockSize side length of a cubic grid cell
* @param d_vorticity buffer with turbulence parameters
* @param d_vel buffer with previous velocity
* @param d_ccvel buffer for locally averaged velocity
* 
*/
__global__ void computeVorticity(int3 gridCount, float blockSize, float3* d_vorticity, float3* d_vel, float3* d_ccvel);

/**
* @brief Updates velocity (wind speed) due to advection in the fluid
* 
* @param gridCount of simulation space in x, y, z directions
* @param gridSize of entire simulation space in x, y, z directions
* @param blockSize side length of a cubic grid cell
* @param d_temp buffer with previous temperature values
* @param d_vel buffer for new velocity values
* @param d_oldvel buffer with previous temperature values
* @param d_alpha_m buffer with semi-lagrangian advection solve
* @param d_smokedensity for buoyancy computations
* @param d_vorticity for confinement computations
* @param externalForce any external forces on the grid cell
*/
__global__ void velocityKernel(int3 gridCount, float3 gridSize, float blockSize, float* d_temp, float3* d_vel,
    float3* d_oldvel, float3* d_alpha_m, float* d_smokedensity, float3* d_vorticity, float3 externalForce);

/**
* @brief Updates temperature changes due to advection in the fluid
* 
* @param gridCount of simulation space in x, y, z directions
* @param gridSize of entire simluation space in x, y, z directions
* @param blockSize side length of a cubic grid cell
* @param d_temp buffer for new temperature values
* @param d_oldtemp buffer with previous temperature values
* @param d_vel buffer with updated velocity values
* @param d_alpha_m buffer with semi-lagrangian advection solve
* @param d_lap buffer to store laplacian for previous temperature values
*/
__global__ void tempAdvectionKernel(int3 gridCount, float3 gridSize, float blockSize, float* d_temp, float* d_oldtemp,
    float3* d_vel, float3* d_alpha_m, float* lap);

/**
* @brief Updates smoke density due to advection in the fluid and changes in mass
* 
* @param gridCount of simulation space in x, y, z directions
* @param gridSize of entire simluation space in x, y, z directions
* @param blockSize side length of a cubic grid cell
* @param d_temp buffer with previous temperature values
* @param d_vel buffer with updated velocity values
* @param d_alpha_m buffer with semi-lagrangian advection
* @param d_smoke buffer for updated smoke density
* @param d_oldsmoke buffer with previous smoke density
* @param d_delta_m buffer with change in mass for this grid cell
*/
__global__ void smokeUpdateKernel(int3 gridCount, float3 gridSize, float blockSize, float* d_temp, float3* d_vel, float3* d_alpha_m,
    float* d_smoke, float* d_oldsmoke, float* d_delta_m);

void kernelLauncher(
    int3 gridCount,
    float gridSize,
    float blockSize,
    float* d_temp,
    float* d_oldtemp,
    float3* d_vel,
    float3* d_oldvel,
    float* d_pressure,
    float3* d_ccvel,
    float3* d_vorticity,
    float* d_smokedensity,
    float* d_oldsmokedensity,
    float* d_smokeRadiance,
    float3 externalForce,
    bool sourcesEnabled,
    int activeBuffer, dim3 Ld, BC bc, dim3 M_in, unsigned int slice);


void resetVariables(
    int3 gridCount,
    float gridSize,
    float blockSize,
    float* d_temp,
    float* d_oldtemp,
    float3* d_vel,
    float3* d_oldvel,
    float* d_smokedensity,
    float* d_oldsmokedensity,
    float* d_pressure,
    dim3 Ld, BC bc, dim3 M_in);

#endif

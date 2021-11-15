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
#include "physics.h"

struct uchar4;
// struct BC that contains all the boundary conditions
typedef struct {
    int x, y; // x and y location of pipe center
    float rad; // radius of pipe
    int chamfer; // chamfer
    float t_s, t_a, t_g; // temperatures in pipe, air, ground
} BC;

int blocksNeeded(int N_i, int M_i);
__device__ int flatten(int col, int row, int z);
__device__ int vflatten(int col, int row, int z);
__device__ int flatten(int col, int row, int z, int width, int height, int depth);
__device__ unsigned char clip(int n);

__device__ float3 operator+(const float3& a, const float3& b);
__device__ float3 operator-(const float3& a, const float3& b);
__device__ float3 operator*(const float3& a, const float& b);
__device__ float3 operator*(const float& b, const float3& a);

void kernelLauncher(uchar4* d_out,
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


void resetVariables(float* d_temp,
    float* d_oldtemp,
    float3* d_vel,
    float3* d_oldvel,
    float* d_smokedensity,
    float* d_oldsmokedensity,
    float* d_pressure,
    dim3 Ld, BC bc, dim3 M_in);

#endif

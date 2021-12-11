#pragma once
#include<cuda.h>
#define _USE_MATH_DEFINES // Keep above math.h import
#include <math.h> 
#include <array>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "../Terrain.h"
#include "module-combustion/module.h"
#include "../errors.h"
#include "advection.h"
#include "pressure.h"
#include "smoke_render.h"

// includes go here


// Kernel launching paramters from paper
__device__ const int M_IX = 8;
__device__ const int M_IY = 8;
__device__ const int M_IZ = 8;

// time step
__device__ float DELTA_T = 0.01f;

/**
* TUNABLE PHYSICS PARAMETERS
*/
__device__ float T_AMBIANT = 20.0f;             // ambiant air temperature without regards to heat added by fire
__device__ float P_ATM = 0.0f;                  // atmospheric pressure
__device__ float BUOY_ALPHA = 0.3f;             // smoke density modifier for buoyancy force computation
__device__ float BUOY_BETA = 0.1f;              // temperature modifier for buoyancy force computation
__device__ int SEMILAGRANGIAN_ITERS = 5;        // number of iterations for semi-lagrangian advection solve
__device__ float VORTICITY_EPSILON = 1.f;       // scalar for amount of dissipated energy added back to system
__device__ float TEMPERATURE_ALPHA = 8e-4;      // temperature diffusion coefficient
__device__ float TEMPERATURE_GAMMA = -8e-5;    // radiative cooling coefficient
__device__ int PRESSURE_JACOBI_ITERATIONS = 10; // number of iterations for jacobi pressure solve
__device__ float EVAP = 0.5362f;                // module water to mass ratio
__device__ float SMOKE_MASS = 16.f;             // smoke mass contribution coefficient
__device__ float SMOKE_WATER = 200.f;           // smoke water contribution coefficient
__device__ float TAU = 200.f;                   // temperature change per pass of wood combusted
__device__ int SMOKE_RAY_SQRT_COUNT = 60;
__device__ float SMOKE_EXTINCTION_COEFF = 15e1;
__device__ float SMOKE_ALBEDO = 0.7f;
__device__ float SMOKE_LIGHT_RADIANCE = 5e0;

namespace Simulation
{
    void initSimulation(Terrain* terrain, int3 gridCount);
    void stepSimulation(float dt, int3 gridCount, float3 gridSize, float sideLength, float* d_out);
    void endSimulation();
    void copyBranchesToVBO(float* vbodptr_branches);
}
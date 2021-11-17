#ifndef PRESSURE_H
#define PRESSURE_H

#include <cstdio>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "advection.h"
#include "kernel.h"
#include "../errors.h"

void forceIncompressibility(int3 gridCount, float blockSize, float3 * d_vel, float* d_pressure);
__global__ void resetPressure(int3 gridCount, float* d_pressure);

#endif /* PRESSURE_H */
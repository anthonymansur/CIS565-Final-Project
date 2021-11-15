#ifndef PRESSURE_H
#define PRESSURE_H

#include <cstdio>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "advection.h"
#include "physics.h"
#include "../errors.h"

void forceIncompressibility(float3 * d_vel, float* d_pressure);
__global__ void resetPressure(float* d_pressure);

#endif /* PRESSURE_H */
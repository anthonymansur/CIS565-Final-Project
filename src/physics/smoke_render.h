#ifndef SMOKE_RENDER_H
#define SMOKE_RENDER_H

#include "advection.h"
#include "kernel.h"


#include <math.h>
#include <stdlib.h>
#include <iostream>

__host__ __device__ bool rayGridIntersect(float3 gridSize, float blockSize, const vec3 ray_orig, const vec3 ray_dir, 
                                 int3 * voxel, float * t);
void smokeRender(int3 gridCount, float3 gridSize, float blockSize, dim3 gridSizeK, dim3 M_i, float* d_out, float *d_smokedensity, float *d_smokeRadiance, float totalTime);

#endif /* SMOKE_RENDER_H */

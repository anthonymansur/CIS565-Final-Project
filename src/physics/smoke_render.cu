#include "smoke_render.h"

__device__ unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__ int sidxClip(int idx, int idxMax) {
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__ int sflatten(int col, int row, int z, int width, int height, int depth) {
    return sidxClip(col, width) + sidxClip(row, height) * width + sidxClip(z, depth) * width * height;
}
__device__ int sflatten(int3 gridCount, int col, int row, int z) {
    if (col >= gridCount.x || row >= gridCount.y || z >= gridCount.z || col < 0 || row < 0 || z < 0)
        printf("sflatten oob");
    return sidxClip(col, gridCount.x) + sidxClip(row, gridCount.y) * gridCount.z + sidxClip(z, gridCount.z) * gridCount.x * gridCount.y;
}

__host__ __device__ void swap(float * a, float * b){
    float temp = *b;
    *b = *a;
    *a = temp;
}
__host__ __device__ float posclip(float a){
    return a > 0 ? a : 0;
}
__host__ __device__ bool rayGridIntersect(float3 gridSize, float blockSize, const vec3 ray_orig, const vec3 ray_dir, int3 * voxel, float * t){
    const float m = 0.f;
    float tmin = (m - ray_orig.x()) / ray_dir.x();
    float tmax = (gridSize.x - ray_orig.x()) / ray_dir.x();
    if (tmin > tmax) swap(&tmin, &tmax);

    float tymin = (m - ray_orig.y()) / ray_dir.y();
    float tymax = (gridSize.y - ray_orig.y()) / ray_dir.y();
    if (tymin > tymax) swap(&tymin, &tymax);
    if ((tmin > tymax) || (tymin > tmax)) return false;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;
    float tzmin = (m - ray_orig.z()) / ray_dir.z();
    float tzmax = (gridSize.z - ray_orig.z()) / ray_dir.z();

    if (tzmin > tzmax) swap(&tzmin, &tzmax);
    if ((tmin > tzmax) || (tzmin > tmax)) return false;

    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;
    
    *t = tmin;
    if(*t < 0) return false;
    voxel->x = static_cast<int> (posclip(ray_orig.x() + *t * ray_dir.x()) / blockSize);
    voxel->y = static_cast<int> (posclip(ray_orig.y() + *t * ray_dir.y()) / blockSize);
    voxel->z = static_cast<int> (posclip(ray_orig.z() + *t * ray_dir.z()) / blockSize);
    return true; 

}

__global__ void smokeLightKernel(int3 gridCount, float3 gridSize, float blockSize, vec3 ray_o, vec3 ray_dir, float * d_smoke, float * voxelRadiance){
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    if(k_x >= SMOKE_RAY_SQRT_COUNT || k_y >= SMOKE_RAY_SQRT_COUNT) return;

    vec3 ray_orig = ray_o + vec3(0, (k_x+0.5) * blockSize, (k_y+0.5) * blockSize);
    float ray_transparency = 1.f;
    int3 step;
    step.x = ray_dir.x() > 0 ? 1 : -1;
    step.y = ray_dir.y() > 0 ? 1 : -1;
    step.z = ray_dir.z() > 0 ? 1 : -1;
    int3 voxel; //Current voxel coordinate
    float t; // Initial hit param value
    if(!rayGridIntersect(gridSize, blockSize, ray_orig, ray_dir, &voxel, &t)) return;
    //printf("%d %d %d\n", voxel.x, voxel.y, voxel.z);
    float3 tMax; 
    // DIV by zero handling
    float3 tDelta = {blockSize / ray_dir.x(), blockSize / ray_dir.y(), blockSize / ray_dir.z()};
    while(true){
        const int k = sflatten(gridCount, voxel.x, voxel.y, voxel.z);
        const float voxelTransp = expf(-(SMOKE_EXTINCTION_COEFF*blockSize/d_smoke[k]));
        //voxelRadiance[k] += ray_transparency;
        voxelRadiance[k] += SMOKE_ALBEDO * SMOKE_LIGHT_RADIANCE * voxelTransp * ray_transparency;
        ray_transparency *= 1-voxelTransp;

        tMax.x = (blockSize - fmod((ray_orig + t*ray_dir).x(), blockSize))/ray_dir.x();
        tMax.y = (blockSize - fmod((ray_orig + t*ray_dir).y(), blockSize))/ray_dir.y();
        tMax.z = (blockSize - fmod((ray_orig + t*ray_dir).z(), blockSize))/ray_dir.z();
        if(tMax.x < tMax.y) {
            if(tMax.x < tMax.z) {
                voxel.x += step.x;
                if(voxel.x >= gridCount.x || voxel.x < 0)return; /* outside grid */
                tMax.x += tDelta.x;
            } else {
                voxel.z += step.z;
                if(voxel.z >= gridCount.z || voxel.z < 0)return;
                tMax.z += tDelta.z;
            }
        } else {
            if(tMax.y < tMax.z) {
                voxel.y += step.y;
                if(voxel.y >= gridCount.y || voxel.y < 0)return;
                tMax.y += tDelta.y;
            } else {
                voxel.z += step.z;
                if(voxel.z >= gridCount.z || voxel.z < 0)return;
                tMax.z += tDelta.z;
            }
        }
    __syncthreads();
    }
    
}
__global__ void resetSmokeRadiance(int3 gridCount, float * voxelRadiance){
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x ) || (k_y >= gridCount.y ) || (k_z >= gridCount.z)) return;
    const int k = sflatten(k_x, k_y, k_z, gridCount.x, gridCount.y, gridCount.z);
    
    voxelRadiance[k] = 0.f;
}
__global__ void generateSmokeColorBuffer(int3 gridCount, float blockSize, uchar4* dev_out, const float* d_smoke, float* d_smokeRadiance) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x ) || (k_y >= gridCount.y ) || (k_z >= gridCount.z)) return;
    const int k = sflatten(k_x, k_y, k_z, gridCount.x, gridCount.y, gridCount.z);
    if(isnan(d_smokeRadiance[k]) || isinf(d_smokeRadiance[k])) d_smokeRadiance[k] = 0;
    const unsigned char transparency = clip((int) (expf(-(fabsf(SMOKE_EXTINCTION_COEFF/d_smoke[k]))* blockSize)*255.f));
    const unsigned char intensity = clip((int) (d_smokeRadiance[k]*255.f));
    
    //const unsigned char intensity = clip((int) (d_smoke[k]*255.f));
    for(unsigned int i = 0; i < 4; i++){
        dev_out[4*k+i].x = intensity;
        dev_out[4*k+i].z = intensity;
        dev_out[4*k+i].y = intensity;
        dev_out[4*k+i].w = transparency; // 255 => solid display
    }
}

void smokeRender(int3 gridCount, float3 gridSize, float blockSize, dim3 gridSizeK, dim3 M_i, uchar4* d_out, float *d_smokedensity, float *d_smokeRadiance){
    // Rendering computations
    const dim3 rayBlockSize(8,8);
    const dim3 rayGridSize(blocksNeeded(SMOKE_RAY_SQRT_COUNT, rayBlockSize.x), 
                           blocksNeeded(SMOKE_RAY_SQRT_COUNT, rayBlockSize.y));
    resetSmokeRadiance<<<gridSizeK, M_i>>>(gridCount, d_smokeRadiance);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    smokeLightKernel<<<rayGridSize, rayBlockSize>>>(gridCount, gridSize, blockSize, vec3(SMOKE_LIGHT_POS), vec3(SMOKE_LIGHT_DIR), d_smokedensity, d_smokeRadiance);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    generateSmokeColorBuffer<<<gridSizeK, M_i>>>(gridCount, blockSize, d_out, d_smokedensity, d_smokeRadiance);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

}
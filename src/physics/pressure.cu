#include "pressure.h"

__device__ int pidxClip(int idx, int idxMax) {
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__ int pflatten(int col, int row, int z, int width, int height, int depth) {
    return pidxClip(col, width) + pidxClip(row, height) * width + pidxClip(z, depth) * width * height;
}
__device__ int pflatten(int3 gridCount, int col, int row, int z) {
    if (col >= gridCount.x || row >= gridCount.y || z >= gridCount.z || col < 0 || row < 0 || z < 0)
        printf("pflatten oob");
    return pidxClip(col, gridCount.x) + pidxClip(row, gridCount.y) * gridCount.z + pidxClip(z, gridCount.z) * gridCount.x * gridCount.y;
}
__device__ int pvflatten(int3 gridCount, int col, int row, int z) {
    if (col >= gridCount.x + 1 || row >= gridCount.y + 1 || z >= gridCount.z + 1 || col < 0 || row < 0 || z < 0)
        printf("pvflatten oob");
    return pidxClip(col, gridCount.x + 1) + pidxClip(row, gridCount.y + 1) * (gridCount.z + 1) + pidxClip(z, gridCount.z + 1) * (gridCount.x + 1) * (gridCount.y + 1);
}

__global__ void prepareSystem(int3 gridCount, float blockSize, const int NFLAT, float3* d_vel, float* d_b, float* d_val, int* d_cind, int * d_rptr) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x) || (k_y >= gridCount.y ) || (k_z >= gridCount.z)) return;
    const int k = pflatten(gridCount, k_x, k_y, k_z);
    // Matrix 
    const int offset = 7 * pflatten(gridCount, k_x, k_y, k_z);
    if(k_x > 0 && k_x < gridCount.x - 1 && k_y > 0 && k_y < gridCount.y - 1 && k_z > 0 && k_z < gridCount.z - 1) {
        //B term
        d_b[k] = d_vel[pflatten(gridCount, k_x+1, k_y, k_z)].x - d_vel[pflatten(gridCount, k_x, k_y, k_z)].x + 
                 d_vel[pflatten(gridCount, k_x, k_y+1, k_z)].y - d_vel[pflatten(gridCount, k_x, k_y, k_z)].y + 
                 d_vel[pflatten(gridCount, k_x, k_y, k_z+1)].z - d_vel[pflatten(gridCount, k_x, k_y, k_z)].z ;
        d_b[k] /= blockSize * DELTA_T;
        d_val [offset    ] =  1;
        d_cind[offset    ] = pflatten(gridCount, k_x, k_y, k_z-1);
        d_val [offset + 1] =  1; 
        d_cind[offset + 1] = pflatten(gridCount, k_x, k_y-1, k_z);
        d_val [offset + 2] =  1; 
        d_cind[offset + 2] = pflatten(gridCount, k_x-1, k_y, k_z);
        d_val [offset + 3] = -6; 
        d_cind[offset + 3] = k;
        d_val [offset + 4] =  1; 
        d_cind[offset + 4] = pflatten(gridCount, k_x+1, k_y, k_z);
        d_val [offset + 5] =  1;
        d_cind[offset + 5] = pflatten(gridCount, k_x, k_y+1, k_z);
        d_val [offset + 6] =  1;
        d_cind[offset + 6] = pflatten(gridCount, k_x, k_y, k_z+1);
        
        d_rptr[k] = offset;
    }
    //PRESSURE DIRICHLET BOUNDARY CONDITION
    else {
        d_b[k] = P_ATM;
        // DUMMY VALUES to preserve alignement
        d_cind[offset    ] =  0;//(k+6)%NFLAT;
        d_cind[offset + 1] =  0;//(k+1)%NFLAT;
        d_cind[offset + 2] =  0;//(k+2)%NFLAT;
        d_cind[offset + 3] =  0;//(k+3)%NFLAT;
        d_cind[offset + 4] =  0;//(k+4)%NFLAT;
        d_cind[offset + 5] =  0;//(k+5)%NFLAT;
        d_val [offset    ] =  0;
        d_val [offset + 1] =  0; 
        d_val [offset + 2] =  0; 
        d_val [offset + 3] =  0; 
        d_val [offset + 4] =  0; 
        d_val [offset + 5] =  0; 
        d_val [offset + 6] = 1; 
        d_cind[offset + 6] = k;
        
        d_rptr[k] = offset;
        if(k == NFLAT-1){
            d_rptr[NFLAT] = 7 * NFLAT;
        }
    }
}

__global__ void prepareJacobiMethod(int3 gridCount, float blockSize, float3* d_vel, float* d_f) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x ) || (k_y >= gridCount.y ) || (k_z >= gridCount.z)) return;
    const int k = pflatten(gridCount, k_x, k_y, k_z);
    if(k_x > 0 && k_x < gridCount.x - 1 && k_y > 0 && k_y < gridCount.y - 1 && k_z > 0 && k_z < gridCount.z - 1){
        d_f[k] = d_vel[pvflatten(gridCount, k_x+1, k_y, k_z)].x - d_vel[pvflatten(gridCount, k_x, k_y, k_z)].x + 
                 d_vel[pvflatten(gridCount, k_x, k_y+1, k_z)].y - d_vel[pvflatten(gridCount, k_x, k_y, k_z)].y + 
                 d_vel[pvflatten(gridCount, k_x, k_y, k_z+1)].z - d_vel[pvflatten(gridCount, k_x, k_y, k_z)].z ;
        d_f[k] /= blockSize ;//* dev_Deltat[0];
    }
    else
        d_f[k] = 0;
}

__global__ void jacobiIterations(int3 gridCount, float blockSize, float * d_pressure, float * d_temppressure, float * d_f) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if(k_x == 0 || k_x >= gridCount.x-1  || k_y == 0 || k_y >= gridCount.y-1 || k_z == 0 || k_z >= gridCount.z-1) return;
    const int k = pflatten(gridCount, k_x, k_y, k_z);
    //d_pressure[k] = d_f[k];
    float * d_oldpressure = d_pressure;
    float * d_newpressure = d_temppressure;
    float * tmp;
    for(int i=0; i < PRESSURE_JACOBI_ITERATIONS; i++){
        __syncthreads();
        d_newpressure[k] = d_oldpressure[pflatten(gridCount, k_x, k_y, k_z-1)] +
                           d_oldpressure[pflatten(gridCount, k_x, k_y-1, k_z)] +
                           d_oldpressure[pflatten(gridCount, k_x-1, k_y, k_z)] +
                           d_oldpressure[pflatten(gridCount, k_x+1, k_y, k_z)] +
                           d_oldpressure[pflatten(gridCount, k_x, k_y+1, k_z)] +
                           d_oldpressure[pflatten(gridCount, k_x, k_y, k_z+1)] ;
        d_newpressure[k] -= blockSize * blockSize * d_f[k];
        d_newpressure[k] /= 6;
        tmp = d_newpressure;
        d_newpressure = d_oldpressure;
        d_oldpressure  = tmp;
    }
    d_pressure[k] = d_oldpressure[k];
    //if(fabsf(k_z - GRID_COUNT/2) * fabsf(k_z - GRID_COUNT/2) + 
    //   fabsf(k_y - GRID_COUNT/2) * fabsf(k_y - GRID_COUNT/2) +
    //   fabsf(k_x - GRID_COUNT/2) * fabsf(k_x - GRID_COUNT/2) < GRID_COUNT  *GRID_COUNT / (5*5*25)){
    //    printf("%f\n", d_pressure[k]);
    //}
}
__global__ void resetPressure(int3 gridCount, float* d_pressure){
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x ) || (k_y >= gridCount.y ) || (k_z >= gridCount.z)) return;
    const int k = pflatten(gridCount, k_x, k_y, k_z);
    d_pressure[k] = 0;
}
__global__ void substractPressureGradient(int3 gridCount, float blockSize, float3 * d_vel, float* d_pressure){
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x ) || (k_y >= gridCount.y ) || (k_z >= gridCount.z )) return;
    if(k_x == 0 || k_x >= gridCount.x-1  || k_y == 0 || k_y >= gridCount.y-1 || k_z == 0 || k_z >= gridCount.z-1) return;
    const int k = pvflatten(gridCount, k_x, k_y, k_z);
    const int kcentered = pflatten(gridCount, k_x,k_y,k_z);
    d_vel[k].x -= DELTA_T *  (d_pressure[pflatten(gridCount, k_x+1,k_y,k_z)] - d_pressure[kcentered]) / blockSize;
    d_vel[k].y -= DELTA_T *  (d_pressure[pflatten(gridCount, k_x,k_y+1,k_z)] - d_pressure[kcentered]) / blockSize;
    d_vel[k].z -= DELTA_T *  (d_pressure[pflatten(gridCount, k_x,k_y,k_z+1)] - d_pressure[kcentered]) / blockSize;
}
void forceIncompressibility(int3 gridCount, float blockSize, float3 * d_vel, float* d_pressure){
    // TODO: CHOLESKI PREPROCESS
    const int NFLAT = gridCount.x * gridCount.y * gridCount.z;
    const dim3 gridSize(blocksNeeded(gridCount.x, M_IX), 
                        blocksNeeded(gridCount.y, M_IY), 
                        blocksNeeded(gridCount.z, M_IZ));
    const dim3 M_i(M_IX, M_IY, M_IZ);

    float * d_f;
    HANDLE_ERROR(cudaMalloc(&d_f, NFLAT*sizeof(float)));
    resetPressure<<<gridSize, M_i>>>(gridCount, d_pressure);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());
    float * d_pressure1;    
    HANDLE_ERROR(cudaMalloc(&d_pressure1, NFLAT*sizeof(float)));
    resetPressure<<<gridSize, M_i>>>(gridCount, d_pressure1);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());
    
    prepareJacobiMethod<<<gridSize, M_i>>>(gridCount, blockSize, d_vel, d_f);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    jacobiIterations<<<gridSize, M_i>>>(gridCount, blockSize, d_pressure, d_pressure1, d_f);
    HANDLE_ERROR(cudaPeekAtLastError());    
    HANDLE_ERROR(cudaDeviceSynchronize());
    substractPressureGradient<<<gridSize, M_i>>>(gridCount, blockSize, d_vel, d_pressure);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaFree(d_f));
    HANDLE_ERROR(cudaFree(d_pressure1));
}
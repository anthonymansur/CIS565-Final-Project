#include "pressure.h"

__device__ int pidxClip(int idx, int idxMax) {
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__ int pflatten(int col, int row, int z, int width, int height, int depth) {
    return pidxClip(col, width) + pidxClip(row, height) * width + pidxClip(z, depth) * width * height;
}
__device__ int pflatten(int col, int row, int z) {
    if (col >= GRID_COUNT_X || row >= GRID_COUNT_Y || z >= GRID_COUNT_Z || col < 0 || row < 0 || z < 0)
        printf("pflatten oob");
    return pidxClip(col, GRID_COUNT_X) + pidxClip(row, GRID_COUNT_Y) * GRID_COUNT_Z + pidxClip(z, GRID_COUNT_Z) * GRID_COUNT_X * GRID_COUNT_Y;
}
__device__ int pvflatten(int col, int row, int z) {
    if (col >= GRID_COUNT_X + 1 || row >= GRID_COUNT_Y + 1 || z >= GRID_COUNT_Z + 1 || col < 0 || row < 0 || z < 0)
        printf("pvflatten oob");
    return pidxClip(col, GRID_COUNT_X + 1) + pidxClip(row, GRID_COUNT_Y + 1) * (GRID_COUNT_Z + 1) + pidxClip(z, GRID_COUNT_Z + 1) * (GRID_COUNT_X + 1) * (GRID_COUNT_Y + 1);
}

__device__ int valIndex(int k_x, int k_y, int k_z){
    const int center_nnz = 7;
    const int boundary_nnz = 1;
    //INTERMEDIATE FACE SIZE
    const int cfnnz = 4*(GRID_COUNT_X - 1) * boundary_nnz + (GRID_COUNT_Y - 2)*(GRID_COUNT_Z - 2) * center_nnz;

    if(k_z == 0){
        return (k_x + k_y * GRID_COUNT_Y) * boundary_nnz;
    }
    else if(k_z == GRID_COUNT_Z - 1){
        return GRID_COUNT_X * GRID_COUNT_Y * boundary_nnz +
               (GRID_COUNT_Z - 2) * cfnnz +
               (k_x + k_y * GRID_COUNT_Y) * boundary_nnz;
    }
    else {
        if(k_y == 0){
            return GRID_COUNT_X * GRID_COUNT_Y * boundary_nnz + 
                   (k_z - 1) * cfnnz + 
                   k_x * boundary_nnz;
        }
        else if(k_y == GRID_COUNT_Y - 1){
            return GRID_COUNT_X * GRID_COUNT_Y * boundary_nnz + 
                   (k_z - 1) * cfnnz + 
                   GRID_COUNT_X * boundary_nnz +
                   (GRID_COUNT_X - 2) * (2 * boundary_nnz + (GRID_COUNT_X - 2)*center_nnz) +
                   (k_x) * boundary_nnz;
        }
        else{
            if(k_x == 0){
                return GRID_COUNT_X * GRID_COUNT_Y * boundary_nnz +
                       (k_z - 1) * cfnnz +
                       GRID_COUNT_X * boundary_nnz +
                       (k_y - 1) * (2*boundary_nnz + (GRID_COUNT_X - 2)*center_nnz);
            }
            else if(k_x == GRID_COUNT_X - 1){
                return GRID_COUNT_X * GRID_COUNT_Y * boundary_nnz +
                       (k_z - 1) * cfnnz +
                       GRID_COUNT_X * boundary_nnz +
                       (k_y - 1) * (2*boundary_nnz + (GRID_COUNT_X-2)*center_nnz)+
                       boundary_nnz + (GRID_COUNT_X-2)*center_nnz;
            }
            else {
                return GRID_COUNT_X * GRID_COUNT_Y * boundary_nnz +
                       (k_z - 1) * cfnnz +
                       GRID_COUNT_X * boundary_nnz +
                       (k_y - 1) * (2*boundary_nnz + (GRID_COUNT_X-2)*center_nnz)+
                       boundary_nnz + (k_x-1) * center_nnz;
            }
        }
    }
}

__global__ void prepareSystem(const int NFLAT, float3* d_vel, float* d_b, float* d_val, int* d_cind, int * d_rptr) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= GRID_COUNT_X) || (k_y >= GRID_COUNT_Y ) || (k_z >= GRID_COUNT_Z)) return;
    const int k = pflatten(k_x, k_y, k_z);
    // Matrix 
    const int offset = 7 * pflatten(k_x, k_y, k_z);
    if(k_x > 0 && k_x < GRID_COUNT_X - 1 && k_y > 0 && k_y < GRID_COUNT_Y - 1 && k_z > 0 && k_z < GRID_COUNT_Z - 1) {
        //B term
        d_b[k] = d_vel[pflatten(k_x+1, k_y, k_z)].x - d_vel[pflatten(k_x, k_y, k_z)].x + 
                 d_vel[pflatten(k_x, k_y+1, k_z)].y - d_vel[pflatten(k_x, k_y, k_z)].y + 
                 d_vel[pflatten(k_x, k_y, k_z+1)].z - d_vel[pflatten(k_x, k_y, k_z)].z ;
        d_b[k] /= BLOCK_SIZE * DELTA_T;
        d_val [offset    ] =  1;
        d_cind[offset    ] = pflatten(k_x, k_y, k_z-1);
        d_val [offset + 1] =  1; 
        d_cind[offset + 1] = pflatten(k_x, k_y-1, k_z);
        d_val [offset + 2] =  1; 
        d_cind[offset + 2] = pflatten(k_x-1, k_y, k_z);
        d_val [offset + 3] = -6; 
        d_cind[offset + 3] = k;
        d_val [offset + 4] =  1; 
        d_cind[offset + 4] = pflatten(k_x+1, k_y, k_z);
        d_val [offset + 5] =  1; 
        d_cind[offset + 5] = pflatten(k_x, k_y+1, k_z);
        d_val [offset + 6] =  1; 
        d_cind[offset + 6] = pflatten(k_x, k_y, k_z+1);
        
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

__global__ void prepareJacobiMethod(float3* d_vel, float* d_f) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= GRID_COUNT_X ) || (k_y >= GRID_COUNT_Y ) || (k_z >= GRID_COUNT_Z)) return;
    const int k = pflatten(k_x, k_y, k_z);
    if(k_x > 0 && k_x < GRID_COUNT_X - 1 && k_y > 0 && k_y < GRID_COUNT_Y - 1 && k_z > 0 && k_z < GRID_COUNT_Z - 1){
        d_f[k] = d_vel[pvflatten(k_x+1, k_y, k_z)].x - d_vel[pvflatten(k_x, k_y, k_z)].x + 
                 d_vel[pvflatten(k_x, k_y+1, k_z)].y - d_vel[pvflatten(k_x, k_y, k_z)].y + 
                 d_vel[pvflatten(k_x, k_y, k_z+1)].z - d_vel[pvflatten(k_x, k_y, k_z)].z ;
        d_f[k] /= BLOCK_SIZE ;//* dev_Deltat[0];
    }
    else
        d_f[k] = 0;
}

__global__ void jacobiIterations(float * d_pressure, float * d_temppressure, float * d_f) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if(k_x == 0 || k_x >= GRID_COUNT_X-1  || k_y == 0 || k_y >= GRID_COUNT_Y-1 || k_z == 0 || k_z >= GRID_COUNT_Z-1) return;
    const int k = pflatten(k_x, k_y, k_z);
    //d_pressure[k] = d_f[k];
    float * d_oldpressure = d_pressure;
    float * d_newpressure = d_temppressure;
    float * tmp;
    for(int i=0; i < PRESSURE_JACOBI_ITERATIONS; i++){
        __syncthreads();
        d_newpressure[k] = d_oldpressure[pflatten(k_x, k_y, k_z-1)] +
                           d_oldpressure[pflatten(k_x, k_y-1, k_z)] +
                           d_oldpressure[pflatten(k_x-1, k_y, k_z)] +
                           d_oldpressure[pflatten(k_x+1, k_y, k_z)] +
                           d_oldpressure[pflatten(k_x, k_y+1, k_z)] +
                           d_oldpressure[pflatten(k_x, k_y, k_z+1)] ;
        d_newpressure[k] -= BLOCK_SIZE * BLOCK_SIZE * d_f[k];
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
__global__ void resetPressure(float* d_pressure){
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= GRID_COUNT_X ) || (k_y >= GRID_COUNT_Y ) || (k_z >= GRID_COUNT_Z)) return;
    const int k = pflatten(k_x, k_y, k_z);
    d_pressure[k] = 0;
}
__global__ void substractPressureGradient(float3 * d_vel, float* d_pressure){
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= GRID_COUNT_X ) || (k_y >= GRID_COUNT_Y ) || (k_z >= GRID_COUNT_Z )) return;
    if(k_x == 0 || k_x >= GRID_COUNT_X-1  || k_y == 0 || k_y >= GRID_COUNT_Y-1 || k_z == 0 || k_z >= GRID_COUNT_Z-1) return;
    const int k = pvflatten(k_x, k_y, k_z);
    const int kcentered = pflatten(k_x,k_y,k_z);
    d_vel[k].x -= DELTA_T *  (d_pressure[pflatten(k_x+1,k_y,k_z)] - d_pressure[kcentered]) / BLOCK_SIZE;
    d_vel[k].y -= DELTA_T *  (d_pressure[pflatten(k_x,k_y+1,k_z)] - d_pressure[kcentered]) / BLOCK_SIZE;
    d_vel[k].z -= DELTA_T *  (d_pressure[pflatten(k_x,k_y,k_z+1)] - d_pressure[kcentered]) / BLOCK_SIZE;
}
void forceIncompressibility(float3 * d_vel, float* d_pressure){
    // TODO: CHOLESKI PREPROCESS
    const int NFLAT = GRID_COUNT_X * GRID_COUNT_Y * GRID_COUNT_Z;
    const dim3 gridSize(blocksNeeded(GRID_COUNT_X, M_IX), 
                        blocksNeeded(GRID_COUNT_Y, M_IY), 
                        blocksNeeded(GRID_COUNT_Z, M_IZ));
    const dim3 M_i(M_IX, M_IY, M_IZ);

    float * d_f;
    HANDLE_ERROR(cudaMalloc(&d_f, NFLAT*sizeof(float)));
    resetPressure<<<gridSize, M_i>>>(d_pressure);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());
    float * d_pressure1;    
    HANDLE_ERROR(cudaMalloc(&d_pressure1, NFLAT*sizeof(float)));
    resetPressure<<<gridSize, M_i>>>(d_pressure1);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());
    
    prepareJacobiMethod<<<gridSize, M_i>>>(d_vel, d_f);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    jacobiIterations<<<gridSize, M_i>>>(d_pressure, d_pressure1, d_f);
    HANDLE_ERROR(cudaPeekAtLastError());    
    HANDLE_ERROR(cudaDeviceSynchronize());
    substractPressureGradient<<<gridSize, M_i>>>(d_vel, d_pressure);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaFree(d_f));
    HANDLE_ERROR(cudaFree(d_pressure1));
}
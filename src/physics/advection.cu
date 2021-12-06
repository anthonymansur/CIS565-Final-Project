/* advection.cu
 * 3-dim. Laplace eq. (heat eq.) by finite difference with shared memory
 * Adapated from Ernest Yeung's implementation  ernestyalumni@gmail.com
 * 2016.07.29
 */
#include "advection.h"

#define RAD 1 // radius of the stencil; helps to deal with "boundary conditions" at (thread) block's ends

int blocksNeeded(int N_i, int M_i) { return (N_i + M_i - 1) / M_i; }

__device__ int idxClip(int idx, int idxMax) {
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__ int flatten(int col, int row, int z, int width, int height, int depth) {
    return idxClip(col, width) + idxClip(row, height) * width + idxClip(z, depth) * width * height;
}

__device__ int flatten(int3 gridCount, int col, int row, int z) {
    if (col >= gridCount.x || row >= gridCount.y || z >= gridCount.z || col < 0 || row < 0 || z < 0)
        printf("flatten oob");
    return idxClip(col, gridCount.x) + idxClip(row, gridCount.y) * gridCount.z + idxClip(z, gridCount.z) * gridCount.x * gridCount.y;
}

__device__ int vflatten(int3 gridCount, int col, int row, int z) {
    if (col >= gridCount.x + 1 || row >= gridCount.y + 1 || z >= gridCount.z + 1 || col < 0 || row < 0 || z < 0)
        printf("vflatten oob");
    return idxClip(col, gridCount.x + 1) + idxClip(row, gridCount.y + 1) * (gridCount.z + 1) + idxClip(z, gridCount.z + 1) * (gridCount.x + 1) * (gridCount.y + 1);
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator*(const float& b, const float3& a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ int d_abs(int a) {
    return a > 0 ? a : -a;
}

__global__ void resetKernelCentered(int3 gridCount, float* d_temp, float* d_oldtemp, float* d_smokedensity, float* d_oldsmokedensity) {
    const int k_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int k_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((k_x >= gridCount.x) || (k_y >= gridCount.y) || (k_z >= gridCount.z)) return;

    const int k = flatten(gridCount, k_x, k_y, k_z);
    d_temp[k] = d_oldtemp[k] = T_AMBIANT;
    d_smokedensity[k] = d_oldsmokedensity[k] = 0.f;
    if (d_abs(k_z - gridCount.z / 2) * d_abs(k_z - gridCount.z / 2) +
        d_abs(k_y - gridCount.y / 2) * d_abs(k_y - gridCount.y / 2) +
        d_abs(k_x - gridCount.x / 2) * d_abs(k_x - gridCount.x / 2) < gridCount.x * gridCount.y / (5 * 5 * 25)) {
        d_oldsmokedensity[k] = d_smokedensity[k] = 1.f;
        d_oldtemp[k] = d_temp[k] = T_AMBIANT + 50.f;
    }
}
__global__ void resetKernelVelocity(int3 gridCount, float3* d_vel, float3* d_oldvel) {
    const int k_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int k_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((k_x >= gridCount.x + 1) || (k_y >= gridCount.y + 1) || (k_z >= gridCount.z + 1)) return;

    const int k = vflatten(gridCount, k_x, k_y, k_z);
    d_vel[k] = make_float3(0.f, 0.f, 0.f);
    d_oldvel[k] = make_float3(0.f, 0.f, 0.f);
}


__device__ float3 getAlpham(int3 gridCount, float3 gridSize, float blockSize, float3* d_vel, float3 pos, int k) {
    // Iteratively compute alpha_m
    float3 alpha_m = d_vel[k] * DELTA_T;
    for (unsigned int i = 0; i < SEMILAGRANGIAN_ITERS; i++) {
        float3 estimated = pos - alpha_m;
        if (estimated.x < blockSize) estimated.x = blockSize;
        if (estimated.y < blockSize) estimated.y = blockSize;
        if (estimated.z < blockSize) estimated.z = blockSize;
        if (estimated.x > gridSize.x - blockSize) estimated.x = gridSize.x - blockSize;
        if (estimated.y > gridSize.y - blockSize) estimated.y = gridSize.y - blockSize;
        if (estimated.z > gridSize.z - blockSize) estimated.z = gridSize.z - blockSize;
        uint3 b = { static_cast<unsigned int>(estimated.x / blockSize),
                   static_cast<unsigned int>(estimated.y / blockSize),
                   static_cast<unsigned int>(estimated.z / blockSize) };
        float3 localCoord = (estimated - make_float3(b.x * blockSize, b.y * blockSize, b.z * blockSize)) * (1 / blockSize);
        alpha_m.x = (1 - localCoord.x) * d_vel[vflatten(gridCount, b.x, b.y, b.z)].x +
            (localCoord.x) * d_vel[vflatten(gridCount, b.x + 1, b.y, b.z)].x;
        alpha_m.y = (1 - localCoord.y) * d_vel[vflatten(gridCount, b.x, b.y, b.z)].y +
            (localCoord.y) * d_vel[vflatten(gridCount, b.x, b.y + 1, b.z)].y;
        alpha_m.z = (1 - localCoord.z) * d_vel[vflatten(gridCount, b.x, b.y, b.z)].z +
            (localCoord.z) * d_vel[vflatten(gridCount, b.x, b.y, b.z + 1)].z;
        alpha_m = alpha_m * DELTA_T;
    }
    //CLIPPING ON FACES
    return alpha_m;
}

__device__ float3 fbuoyancy(int3 gridCount, float* d_smoke, float* d_temp, int k_x, int k_y, int k_z) {
    const int k = flatten(gridCount, k_x, k_y, k_z);
    float3 f = make_float3(0, 0, 0);
    if (k_y == gridCount.y - 1) {
        f.y += -BUOY_ALPHA * d_smoke[k];
        f.y += BUOY_BETA * (d_temp[k] - T_AMBIANT);
    }
    else {
        f.y += -0.5 * BUOY_ALPHA * (d_smoke[k] + d_smoke[flatten(gridCount, k_x, k_y + 1, k_z)]);
        f.y += BUOY_BETA * ((d_temp[k] + d_temp[flatten(gridCount, k_x, k_y + 1, k_z)]) * 0.5f - T_AMBIANT);
    }
    return f;
}

__device__ float3 fconfinement(int3 gridCount, float blockSize, float3* d_vorticity, int k_x, int k_y, int k_z) {
    const int k = flatten(gridCount, k_x, k_y, k_z);
    if (k_x == gridCount.x - 1 || k_y == gridCount.y - 1 || k_z == gridCount.z - 1)
        return make_float3(0, 0, 0);

    vec3 N(vec3(d_vorticity[flatten(gridCount, k_x + 1, k_y, k_z)]).length() - vec3(d_vorticity[k]).length(),
        vec3(d_vorticity[flatten(gridCount, k_x, k_y + 1, k_z)]).length() - vec3(d_vorticity[k]).length(),
        vec3(d_vorticity[flatten(gridCount, k_x, k_y, k_z + 1)]).length() - vec3(d_vorticity[k]).length());
    if (N.length() < 1e-6) return make_float3(0, 0, 0);
    N.make_unit_vector();
    vec3 f = VORTICITY_EPSILON * blockSize * cross(N, vec3(d_vorticity[k]));
    return f.toFloat3();
}

__global__ void computeVorticity(int3 gridCount, float blockSize, float3* d_vorticity, float3* d_vel, float3* d_ccvel) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((k_x >= gridCount.x) || (k_y >= gridCount.y) || (k_z >= gridCount.z)) return;

    const int k = flatten(gridCount, k_x, k_y, k_z);
    if (!k_x || k_x == gridCount.x - 1 || !k_y || k_y == gridCount.y - 1 || !k_z || k_z == gridCount.z - 1) {
        d_vorticity[k] = make_float3(0, 0, 0);
        return;
    }

    d_ccvel[k].x = (d_vel[vflatten(gridCount, k_x, k_y, k_z)].x + d_vel[vflatten(gridCount, k_x + 1, k_y, k_z)].x) * 0.5f;
    d_ccvel[k].y = (d_vel[vflatten(gridCount, k_x, k_y, k_z)].y + d_vel[vflatten(gridCount, k_x, k_y + 1, k_z)].y) * 0.5f;
    d_ccvel[k].x = (d_vel[vflatten(gridCount, k_x, k_y, k_z)].z + d_vel[vflatten(gridCount, k_x, k_y, k_z + 1)].z) * 0.5f;

    __syncthreads();

    d_vorticity[k].x = d_ccvel[flatten(gridCount, k_x, k_y + 1, k_z)].z - d_ccvel[flatten(gridCount, k_x, k_y - 1, k_z)].z -
        d_ccvel[flatten(gridCount, k_x, k_y, k_z + 1)].y + d_ccvel[flatten(gridCount, k_x, k_y, k_z + 1)].y;
    d_vorticity[k].x /= 2 * blockSize;

    d_vorticity[k].y = d_ccvel[flatten(gridCount, k_x, k_y, k_z + 1)].x - d_ccvel[flatten(gridCount, k_x, k_y, k_z - 1)].x -
        d_ccvel[flatten(gridCount, k_x + 1, k_y, k_z)].z + d_ccvel[flatten(gridCount, k_x - 1, k_y, k_z)].z;
    d_vorticity[k].y /= 2 * blockSize;

    d_vorticity[k].z = d_ccvel[flatten(gridCount, k_x + 1, k_y, k_z)].y - d_ccvel[flatten(gridCount, k_x - 1, k_y, k_z)].y -
        d_ccvel[flatten(gridCount, k_x, k_y + 1, k_z)].x + d_ccvel[flatten(gridCount, k_x, k_y - 1, k_z)].x;
    d_vorticity[k].z /= 2 * blockSize;
}

__global__ void sourceskernel(int3 gridCount, float* d_smokedensity, float* d_temp) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((k_x >= gridCount.x) || (k_y >= gridCount.y) || (k_z >= gridCount.z)) return;

    const int k = flatten(gridCount, k_x, k_y, k_z);
    if (d_abs(k_z - gridCount.x / 2) * d_abs(k_z - gridCount.x / 2) +
        d_abs(k_y - gridCount.y / 2) * d_abs(k_y - gridCount.y / 2) +
        d_abs(k_x - gridCount.z / 2) * d_abs(k_x - gridCount.z / 2) < gridCount.x * gridCount.x / (7 * 25)) {
        d_smokedensity[k] = 1.5;
        d_temp[k] = T_AMBIANT + 100.f;
    }
}

__global__ void velocityKernel(int3 gridCount, float3 gridSize, float blockSize, float* d_temp, float3* d_vel, 
    float3* d_oldvel, float3* d_alpha_m, float* d_smokedensity, float3* d_vorticity, float3 externalForce) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x) || (k_y >= gridCount.y) || (k_z >= gridCount.z)) return;
    const int k = flatten(gridCount, k_x, k_y, k_z);

    // External forces
    float3 f = { 0, 0, 0 };
    float3 fext = { -0.,0.15,0. };
    f = f + fext + externalForce;
    f = f + fconfinement(gridCount, blockSize, d_vorticity, k_x, k_y, k_z);
    f = f + fbuoyancy(gridCount, d_smokedensity, d_temp, k_x, k_y, k_z);

    //Boundary conditions
    if (k_x == 0 || k_x == gridCount.x - 1)
        d_vel[k].x = 0;
    if (k_y == 0 || k_y == gridCount.y - 1)
        d_vel[k].y = 0;
    if (k_z == 0 || k_z == gridCount.z - 1)
        d_vel[k].z = 0;

    // Semi Lagrangian Advection
    float3 pos = make_float3((k_x + 0.5f) * blockSize, (k_y + 0.5f) * blockSize, (k_z + 0.5f) * blockSize);
    float3 alpha_m = getAlpham(gridCount, gridSize, blockSize, d_oldvel, pos, k);
    d_alpha_m[k] = alpha_m;

    // Backtracing 
    float3 estimated = pos - 2 * alpha_m;

    //Clip on boundaries faces
    if (estimated.x < blockSize) estimated.x = blockSize;
    if (estimated.y < blockSize) estimated.y = blockSize;
    if (estimated.z < blockSize) estimated.z = blockSize;
    if (estimated.x > gridSize.x - blockSize) estimated.x = gridSize.x - blockSize;
    if (estimated.y > gridSize.y - blockSize) estimated.y = gridSize.y - blockSize;
    if (estimated.z > gridSize.z - blockSize) estimated.z = gridSize.z - blockSize;

    int3 b = { static_cast<int>(estimated.x / blockSize),
               static_cast<int>(estimated.y / blockSize),
               static_cast<int>(estimated.z / blockSize) };

    float3 localCoord = (estimated - make_float3(b.x * blockSize, b.y * blockSize, b.z * blockSize)) * (1 / blockSize);

    //Velocity per component
    float3 dv;
    dv.x = (1 - localCoord.x) * d_oldvel[vflatten(gridCount, b.x, b.y, b.z)].x +
        (localCoord.x) * d_oldvel[vflatten(gridCount, b.x + 1, b.y, b.z)].x;
    dv.y = (1 - localCoord.y) * d_oldvel[vflatten(gridCount, b.x, b.y, b.z)].y +
        (localCoord.y) * d_oldvel[vflatten(gridCount, b.x, b.y + 1, b.z)].y;
    dv.z = (1 - localCoord.z) * d_oldvel[vflatten(gridCount, b.x, b.y, b.z)].z +
        (localCoord.z) * d_oldvel[vflatten(gridCount, b.x, b.y, b.z + 1)].z;
    d_vel[k] = dv + f * DELTA_T;    
}

__device__ float scalarLinearInt(int3 gridCount, float blockSize, float* scalarfield, float3 pos, float oobvalue) {
    // trilinear interpolation
    int x = static_cast<int> (pos.x / blockSize);
    int y = static_cast<int> (pos.y / blockSize);
    int z = static_cast<int> (pos.z / blockSize);

    //getting voxel edges
    if (fabs(pos.x - x * blockSize) > fabs(pos.x - (x + 1) * blockSize)) x++;
    if (fabs(pos.y - y * blockSize) > fabs(pos.y - (y + 1) * blockSize)) y++;
    if (fabs(pos.z - z * blockSize) > fabs(pos.z - (z + 1) * blockSize)) z++;

    //pos is inside voxels [x-1, x] [y-1, y] [z-1, z]
    //bound check
    if (x <= 0 || x >= gridCount.x || y <= 0 || y >= gridCount.y || z <= 0 || z >= gridCount.z)
        return oobvalue;

    float tx = (pos.x / blockSize - (x - 0.5f));
    float ty = (pos.y / blockSize - (y - 0.5f));
    float tz = (pos.z / blockSize - (z - 0.5f));

    // bottom z then upper z
    float bybz = tx * scalarfield[flatten(gridCount, x, y - 1, z - 1)] + (1 - tx) * scalarfield[flatten(gridCount, x - 1, y - 1, z - 1)];
    float uybz = tx * scalarfield[flatten(gridCount, x, y, z - 1)] + (1 - tx) * scalarfield[flatten(gridCount, x - 1, y, z - 1)];
    float bz = (1 - ty) * bybz + ty * uybz;
    float byuz = tx * scalarfield[flatten(gridCount, x, y - 1, z)] + (1 - tx) * scalarfield[flatten(gridCount, x - 1, y - 1, z)];
    float uyuz = tx * scalarfield[flatten(gridCount, x, y, z)] + (1 - tx) * scalarfield[flatten(gridCount, x - 1, y, z)];
    float uz = (1 - ty) * byuz + ty * uyuz;
    return (1 - tz) * bz + tz * uz;
}

__device__ float laplacian(int3 gridCount, float blockSize, float* field, float oobvalue, int k_x, int k_y, int k_z) {
    float out = 0;
    out += -6 * field[flatten(gridCount, k_x, k_y, k_z)];
    out += k_x < gridCount.x - 1 ? field[flatten(gridCount, k_x + 1, k_y, k_z)] : oobvalue;
    out += k_y < gridCount.y - 1 ? field[flatten(gridCount, k_x, k_y + 1, k_z)] : oobvalue;
    out += k_z < gridCount.z - 1 ? field[flatten(gridCount, k_x, k_y, k_z + 1)] : oobvalue;
    out += k_x > 0 ? field[flatten(gridCount, k_x - 1, k_y, k_z)] : oobvalue;
    out += k_y > 0 ? field[flatten(gridCount, k_x, k_y - 1, k_z)] : oobvalue;
    out += k_z > 0 ? field[flatten(gridCount, k_x, k_y, k_z - 1)] : oobvalue;
    out /= blockSize * blockSize;
    return out;
}
__global__ void tempAdvectionKernel(int3 gridCount, float3 gridSize, float blockSize, float* d_temp, float* d_oldtemp, 
    float3* d_vel, float3* d_alpha_m, float* lap, float* d_deltaM) {
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x) || (k_y >= gridCount.y) || (k_z >= gridCount.z)) return;
    const int k = flatten(k_x, k_y, k_z, gridCount.x, gridCount.y, gridCount.z);

    // Advection
    float3 pos = make_float3((k_x + 0.5f) * blockSize, (k_y + 0.5f) * blockSize, (k_z + 0.5f) * blockSize);
    float3 alpha_m = d_alpha_m[k];
    d_alpha_m[k] = alpha_m;

    // Backtracing 
    float3 estimated = pos - 2 * alpha_m;

//
//# if __CUDA_ARCH__>=200
//    printf("Estimated = x: %f, y: %f, z: %f\n", estimated.x, estimated.y, estimated.z);
//    printf("pos = x: %f, y: %f, z: %f\n", pos.x, pos.y, pos.z);
//    printf("blockSize = %f\n", blockSize);
//#endif 

    // Clipping
    if (estimated.x < blockSize) estimated.x = blockSize;
    if (estimated.y < blockSize) estimated.y = blockSize;
    if (estimated.z < blockSize) estimated.z = blockSize;
    if (estimated.x > gridSize.x - blockSize) estimated.x = gridSize.x - blockSize;
    if (estimated.y > gridSize.y - blockSize) estimated.y = gridSize.y - blockSize;
    if (estimated.z > gridSize.z - blockSize) estimated.z = gridSize.z - blockSize;

    float dt = scalarLinearInt(gridCount, blockSize, d_oldtemp, estimated, T_AMBIANT);
    estimated = pos - alpha_m;

    //Clip on boundaries faces
    if (estimated.x < blockSize) estimated.x = blockSize;
    if (estimated.y < blockSize) estimated.y = blockSize;
    if (estimated.z < blockSize) estimated.z = blockSize;
    if (estimated.x > gridSize.x - blockSize) estimated.x = gridSize.x - blockSize;
    if (estimated.y > gridSize.y - blockSize) estimated.y = gridSize.y - blockSize;
    if (estimated.z > gridSize.z - blockSize) estimated.z = gridSize.z - blockSize;

    float dtR = TEMPERATURE_GAMMA * powf(scalarLinearInt(gridCount, blockSize, d_oldtemp, estimated, T_AMBIANT) - T_AMBIANT, 4);
    lap[k] = laplacian(gridCount, blockSize, d_oldtemp, T_AMBIANT, k_x, k_y, k_z);

    __syncthreads();

    dtR += TEMPERATURE_ALPHA * scalarLinearInt(gridCount, blockSize, lap, estimated, 0);

    // mass contribution
    float dtm = TAU * d_deltaM[k];
    if (dtm > 0) dtm *= -1.0f; // temporary fix for positive change in mass

    d_temp[k] = -dtm + dt + dtR * 2 * DELTA_T;
    # if __CUDA_ARCH__>=200
    //if (d_temp[k] != 20.f) {
    //    printf("d_temp[%d] = %f, dtm = %f, dt = %f, dtR = %f\n", k, d_temp[k], dtm, dt, dtR);
    //}
    //if (lap[k] != 0.0f) {
    //    printf("lap[%d] = %f\n", k, lap[k]);
    //}
    //if (d_deltaM[k] != 0.f) {
    //    printf("%f\n", d_deltaM[k]);
    //}
    
       
    #endif 
}

__global__ void smokeUpdateKernel(int3 gridCount, float3 gridSize, float blockSize, float* d_temp, float3* d_vel, float3* d_alpha_m, 
    float* d_smoke, float* d_oldsmoke, float* d_delta_m) {

    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x) || (k_y >= gridCount.y) || (k_z >= gridCount.z)) return;
    const int k = flatten(k_x, k_y, k_z, gridCount.x, gridCount.y, gridCount.z);
    // Advection
    float3 pos = make_float3((k_x + 0.5f) * blockSize, (k_y + 0.5f) * blockSize, (k_z + 0.5f) * blockSize);
    float3 alpha_m = d_alpha_m[k];

    // Backtracing 
    float3 estimated = pos - 2 * alpha_m;

    //Clip on boundaries faces
    if (estimated.x < blockSize) estimated.x = blockSize;
    if (estimated.y < blockSize) estimated.y = blockSize;
    if (estimated.z < blockSize) estimated.z = blockSize;
    if (estimated.x > gridSize.x - blockSize) estimated.x = gridSize.x - blockSize;
    if (estimated.y > gridSize.y - blockSize) estimated.y = gridSize.y - blockSize;
    if (estimated.z > gridSize.z - blockSize) estimated.z = gridSize.z - blockSize;

    // Contribution to smoke density due to advection of fluid
    float ds = scalarLinearInt(gridCount, blockSize, d_oldsmoke, estimated, 0.f);

    if (d_delta_m[k] > 0) d_delta_m[k] *= -1.0f;

    // Contribution to smoke density due to mass loss and evaporation (d_delta_m is negative)
    ds -= (SMOKE_MASS * d_delta_m[k]) + (EVAP * SMOKE_WATER * d_delta_m[k]);

    __syncthreads();
    //d_smoke[k] = ds;
    d_smoke[k] += 0.1f;
}

void initGridBuffers(
    int3 gridCount,
    float* d_temp,
    float* d_oldtemp,
    float3* d_vel,
    float3* d_oldvel,
    float* d_smokedensity,
    float* d_oldsmokedensity,
    float* d_pressure,
    dim3 M_in) {
    const dim3 gridSizeC(blocksNeeded(gridCount.x, M_in.x),
        blocksNeeded(gridCount.y, M_in.y),
        blocksNeeded(gridCount.z, M_in.z));
    const dim3 gridSizeV(blocksNeeded(gridCount.x + 1, M_in.x),
        blocksNeeded(gridCount.y + 1, M_in.y),
        blocksNeeded(gridCount.z + 1, M_in.z));

    resetKernelCentered << <gridSizeC, M_in >> > (gridCount, d_temp, d_oldtemp, d_smokedensity, d_oldsmokedensity);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());

    resetKernelVelocity << <gridSizeV, M_in >> > (gridCount, d_vel, d_oldvel);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());

    resetPressure << <gridSizeC, M_in >> > (gridCount, d_pressure);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());
}
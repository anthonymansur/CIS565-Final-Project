#ifndef __PHYSICS_H__
#define __PHYSICS_H__
//#include <Windows.h>
#include <cuda.h>
#include <math.h>
 
#include "../errors.h"

#include "advection.h"

// Tunable grid parameters
#define GRID_COUNT_X 60
#define GRID_COUNT_Y 60
#define GRID_COUNT_Z 60
#define GRID_SIZE 1.0f
#define BLOCK_SIZE 1.0f / 60.0f // (GRID_SIZE / GRID_COUNT)
#define M_IX 8
#define M_IY 8
#define M_IZ 8

// Tunable physics parameters
#define DELTA_T 0.05f
#define HEAT_PARAM1 0.005f
#define HEAT_PARAM2 1.0f
#define T_AMBIANT 20.0f
#define TEMPERATURE_ALPHA 8e-5
#define TEMPERATURE_GAMMA -4e-7
#define P_ATM 0.0f
#define PRESSURE_JACOBI_ITERATIONS 10
#define SEMILAGRANGIAN_ITERS 5
#define BUOY_ALPHA 0.3f // SMOKE DENSITY
#define BUOY_BETA 0.1f // TEMPERATURE
#define VORTICITY_EPSILON 1.0f

class Physics
{
private:
    dim3 dev_L3;
    //dev_Grid3d * dev_grid3d;
    //Grid3d * grid3d;
    int activeBuffer = 0;
    bool gridEnabled = false;
    bool raysEnabled = false;
    bool sourcesEnabled = true;
    //float * smokeQuadsPositions;
    //uint * smokeIndexes;
    //float * smokeQuadsColors;
    //GLuint smokeQuadVBO;
    //GLuint smokeQuadIndexBO;
    //float3 externalForce;
    //GLuint smokeColorBufferObj = 0;
    //cudaGraphicsResource *cuda_smokeColorBufferObj_resource;

public:
    Physics();
    ~Physics();
    void initSmokeQuads();
    void renderSmokeQuads(unsigned int cameraAxis);
    void renderGrid();
    void renderLightRays();
    void renderExternalForce();
    inline void toggleGrid() { gridEnabled = !gridEnabled; };
    inline void toggleSources() { sourcesEnabled = !sourcesEnabled; };
    void render(unsigned int cameraAxis);
    void update();
    void reset();
    void addExternalForce(float3 f);
};

#endif


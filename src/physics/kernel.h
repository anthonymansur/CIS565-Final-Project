#pragma once
#include "../Terrain.h"

// includes go here

#define M_IX 8
#define M_IY 8
#define M_IZ 8

#define DELTA_T 0.05f
#define HEAT_PARAM1 0.005f
#define HEAT_PARAM2 1.0f

// tunable physics paramters
#define T_AMBIANT 20.0f
#define P_ATM 0.0f
#define BUOY_ALPHA 0.3f // SMOKE DENSITY
#define BUOY_BETA 0.1f // TEMPERATURE
#define SEMILAGRANGIAN_ITERS 5
#define VORTICITY_EPSILON 1
#define TEMPERATURE_ALPHA 8e-5
#define TEMPERATURE_GAMMA -4e-7
#define PRESSURE_JACOBI_ITERATIONS 10


namespace Simulation
{
    void initSimulation(Terrain* terrain);
    void stepSimulation(float dt);
    void endSimulation();
}
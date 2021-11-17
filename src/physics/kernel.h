#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

// includes go here

namespace Simulation 
{
    void initSimulation(int N);
    void stepSimulation(float dt);
    void copyToVBO(float* vbo_positions);
    void endSimulation();
}
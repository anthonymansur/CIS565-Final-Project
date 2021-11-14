#pragma once
#include "../Terrain.h"

// includes go here

namespace Simulation 
{
    void initSimulation(Terrain* terrain);
    void stepSimulation(float dt);
    void endSimulation();
}
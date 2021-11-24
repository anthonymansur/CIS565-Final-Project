#include <cuda.h>
#define _USE_MATH_DEFINES // Keep above math.h import
#include <math.h> 
#include "kernel.h"
#include "module-combustion/module.h"
#include "../errors.h" // TODO: move file to different location

/*****************
* Configuration *
*****************/

#define blockSize 128

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

// kernel size
int numOfModules;
dim3 threadsPerBlock(blockSize);

// buffers to hold in our graph data
Node* dev_nodes;
Edge* dev_edges;
Module* dev_modules;
ModuleEdge* dev_moduleEdges;

/******************
* initSimulation *
******************/

void Simulation::initSimulation(Terrain* terrain)
{
    numOfModules = terrain->modules.size();
    dim3 fullBlocksPerGrid((numOfModules + blockSize - 1) / blockSize);

    // Allocate buffers for the modules
    HANDLE_ERROR(cudaMalloc((void**)&dev_nodes, terrain->nodes.size() * sizeof(Node)));
    HANDLE_ERROR(cudaMemcpy(dev_nodes, terrain->nodes.data(), terrain->nodes.size(), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_edges, terrain->edges.size() * sizeof(Edge)));
    HANDLE_ERROR(cudaMemcpy(dev_edges, terrain->edges.data(), terrain->edges.size(), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_modules, terrain->modules.size() * sizeof(Module)));
    HANDLE_ERROR(cudaMemcpy(dev_modules, terrain->modules.data(), terrain->modules.size(), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_moduleEdges, terrain->modules.size() * sizeof(ModuleEdge)));
    HANDLE_ERROR(cudaMemcpy(dev_moduleEdges, terrain->moduleEdges.data(), terrain->moduleEdges.size(), cudaMemcpyHostToDevice));

    // TODO: check cuda error

    kernInitModules << <fullBlocksPerGrid, blockSize >> > (numOfModules, dev_nodes, dev_edges, dev_modules);

    // Send back to host to check
    HANDLE_ERROR(cudaMemcpy(terrain->nodes.data(), dev_nodes, terrain->nodes.size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(terrain->edges.data(), dev_edges, terrain->edges.size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(terrain->modules.data(), dev_modules, terrain->modules.size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(terrain->moduleEdges.data(), dev_moduleEdges, terrain->moduleEdges.size(), cudaMemcpyDeviceToHost));
}

/******************
* stepSimulation *
******************/

void Simulation::stepSimulation(float dt)
{
    dim3 fullBlocksPerGrid((numOfModules + blockSize - 1) / blockSize);

    // For each module in the forest
    // - Update mass
    // - Perform radii update
    // - Update temperature
    // - Update released water content
    kernModuleCombustion << <fullBlocksPerGrid, blockSize >> > (dt, numOfModules, dev_nodes, dev_edges, dev_modules);

    // For each grid point x in grid space
    // - update mass and water content
    // TODO: implement

    // Update air temperature
    // update drag forces (wind)
    // update smoke density (qs), water vapor (qv), condensed water (qc),
    // and rain (qc)
    // TODO: implement

    // For each module in the forest
    // cull modules (and their children) that have zero mass
    // TODO: implement
}

/******************
* endSimulation *
******************/
void Simulation::endSimulation()
{
    cudaFree(dev_modules);
    cudaFree(dev_edges);
    cudaFree(dev_nodes);
}
#include <cuda.h>
#include "kernel.h"

/*****************
* Configuration *
*****************/

#define blockSize 128;



/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

// buffers to hold in our graph data
Node* dev_nodes;
Edge* dev_edges;
Module* dev_modules;

/******************
* initSimulation *
******************/
void Simulation::initSimulation(Terrain* terrain)
{
    // Allocate buffers for the modules
    cudaMalloc((void**)&dev_nodes, terrain->nodes.size() * sizeof(Node));
    cudaMemcpy(dev_nodes, terrain->nodes.data(), terrain->nodes.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_edges, terrain->edges.size() * sizeof(Edge));
    cudaMemcpy(dev_edges, terrain->edges.data(), terrain->edges.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_modules, terrain->modules.size() * sizeof(Module));
    cudaMemcpy(dev_modules, terrain->modules.data(), terrain->modules.size(), cudaMemcpyHostToDevice);
    
    // TODO: check cuda error
}

/******************
* stepSimulation *
******************/
void Simulation::stepSimulation(float dt)
{
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
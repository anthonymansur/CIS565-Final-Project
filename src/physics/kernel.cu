#include "kernel.h"

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
float3* dev_temp;
float3* dev_old_temp;

/******************
* initSimulation *
******************/

void Simulation::initSimulation(Terrain* terrain)
{
    numOfModules = terrain->modules.size();
    dim3 fullBlocksPerGrid((numOfModules + blockSize - 1) / blockSize);

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

__global__ void kernModuleCombustion(float time, int N, Node* nodes, Edge* edges, Module* modules) 
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    Module& module = modules[index];
    Node& rootNode = nodes[module.rootNode];

    // TODO: implement
}

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
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
int numOfEdges;
dim3 threadsPerBlock(blockSize);

// buffers to hold in our graph data
Node* dev_nodes;
Edge* dev_edges;
Module* dev_modules;
ModuleEdge* dev_moduleEdges;

// Grid Dimensions
int3 gridCount = { 20, 20, 20 };        
float3 gridSize = { 20.f, 20.f, 20.f };
float sideLength = 1.f; // "blockSize"

// Grid Kernel Launch params
const dim3 M_in(M_IX, M_IY, M_IZ);

// TODO add rest of grid params
float* dev_temp;
float* dev_oldtemp;
float3* dev_vel;
float3* dev_oldvel;
float* dev_pressure;
float3* dev_ccvel;
float3* dev_vorticity;
float* dev_smokedensity;
float* dev_oldsmokedensity;
float* dev_deltaM;

Terrain* m_terrain;

/******************
* initSimulation *
******************/

void Simulation::initSimulation(Terrain* terrain)
{
    m_terrain = terrain;
    numOfModules = terrain->modules.size();
    numOfEdges = terrain->edges.size();
    int numOfGrids = gridCount.x * gridCount.y * gridCount.z;
    dim3 fullBlocksPerGrid((numOfModules + blockSize - 1) / blockSize);

    // Allocate buffers for the modules
    HANDLE_ERROR(cudaMalloc((void**)&dev_nodes, terrain->nodes.size() * sizeof(Node)));
    HANDLE_ERROR(cudaMemcpy(dev_nodes, terrain->nodes.data(), terrain->nodes.size() * sizeof(Node), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_edges, terrain->edges.size() * sizeof(Edge)));
    HANDLE_ERROR(cudaMemcpy(dev_edges, terrain->edges.data(), terrain->edges.size() * sizeof(Edge), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_modules, terrain->modules.size() * sizeof(Module)));
    HANDLE_ERROR(cudaMemcpy(dev_modules, terrain->modules.data(), terrain->modules.size() * sizeof(Module), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_moduleEdges, terrain->moduleEdges.size() * sizeof(ModuleEdge)));
    HANDLE_ERROR(cudaMemcpy(dev_moduleEdges, terrain->moduleEdges.data(), terrain->moduleEdges.size() * sizeof(ModuleEdge), cudaMemcpyHostToDevice));

    // Allocate grid buffers
    HANDLE_ERROR(cudaMalloc((void**)&dev_temp, numOfGrids * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_oldtemp, numOfGrids * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_vel, numOfGrids * sizeof(float3)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_oldvel, numOfGrids * sizeof(float3)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_pressure, numOfGrids * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_ccvel, numOfGrids * sizeof(float3)));
    HANDLE_ERROR(cudaMemset(dev_ccvel, 0, numOfGrids * sizeof(float3)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_vorticity, numOfGrids * sizeof(float3)));
    HANDLE_ERROR(cudaMemset(dev_vorticity, 0, numOfGrids * sizeof(float3)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_smokedensity, numOfGrids * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_oldsmokedensity, numOfGrids * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_deltaM, numOfGrids * sizeof(float)));
    HANDLE_ERROR(cudaMemset(dev_deltaM, 0, numOfGrids * sizeof(float)));

    initGridBuffers(gridCount, dev_temp, dev_oldtemp, dev_vel, dev_oldvel, dev_smokedensity, dev_oldsmokedensity, dev_pressure, M_in);

    kernInitModules << <fullBlocksPerGrid, blockSize >> > (numOfModules, dev_nodes, dev_edges, dev_modules);

    cudaDeviceSynchronize();
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
    kernModuleCombustion << <fullBlocksPerGrid, blockSize >> > (dt, numOfModules, gridCount, sideLength, dev_nodes, dev_edges, dev_modules, dev_moduleEdges, dev_oldtemp);

    // For each grid point x in grid space
    // - update mass and water content
    // TODO: implement
    const dim3 gridDim(blocksNeeded(gridCount.x, M_IX), blocksNeeded(gridCount.y, M_IY), blocksNeeded(gridCount.z, M_IZ));
    
    kernComputeChangeInMass<<<gridDim, M_in>>>(gridCount, numOfModules, sideLength, dev_modules, dev_deltaM);

    // Update air temperature
    // update drag forces (wind)
    // update smoke density (qs), water vapor (qv), condensed water (qc),
    // and rain (qc)
    float* dev_lap;
    float3* dev_alpha_m;
    float3 externalForce = { 0.f, 0.f, 0.f };

    HANDLE_ERROR(cudaMalloc(&dev_lap, gridCount.x * gridCount.y * gridCount.z * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_alpha_m, gridCount.x * gridCount.y * gridCount.z * sizeof(float3)));

    computeVorticity << <gridDim, M_in >> > (gridCount, blockSize, dev_vorticity, dev_oldvel, dev_ccvel);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());

    velocityKernel << <gridDim, M_in >> > (gridCount, gridSize, blockSize, dev_oldtemp, dev_vel, dev_oldvel, dev_alpha_m, dev_oldsmokedensity, dev_vorticity, externalForce);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());

    // Pressure Solve
    forceIncompressibility(gridCount, blockSize, dev_vel, dev_pressure);

    tempAdvectionKernel << <gridDim, M_in >> > (gridCount, gridSize, blockSize, dev_temp, dev_oldtemp, dev_vel, dev_alpha_m, dev_lap, dev_deltaM);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());

    //float* h_temp = (float*)malloc(sizeof(float) * 20 * 20 * 20);
    //cudaMemcpy(h_temp, dev_temp, sizeof(float) * 20 * 20 * 20, cudaMemcpyDeviceToHost);
    //int num = 0;
    //for (int i = 0; i < 20 * 20 * 20; i++) {
    //    //if (h_temp[i] != 20.0f && h_temp[i] != 0.f) printf("index: %d, value: %f\n", i, h_temp[i]);
    //    if (h_temp[i] != 20.0f && h_temp[i] != 0.f) num++;
    //}
    //printf("%d\n", num);
    //free(h_temp);

    smokeUpdateKernel << <gridDim, M_in >> > (gridCount, gridSize, blockSize, dev_oldtemp, dev_vel, dev_alpha_m, dev_smokedensity, 
        dev_oldsmokedensity, dev_deltaM);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaFree(dev_alpha_m));
    HANDLE_ERROR(cudaFree(dev_lap));

    // Ping-pong buffers
    std::swap(dev_temp, dev_oldtemp);
    std::swap(dev_vel, dev_oldvel);
    std::swap(dev_smokedensity, dev_oldsmokedensity);

    // For each module in the forest
    // cull modules (and their children) that have zero mass
    // TODO: implement

    cudaDeviceSynchronize();
}

/****************
* endSimulation *
*****************/
void Simulation::endSimulation()
{
    // Send back to host to check
    HANDLE_ERROR(cudaMemcpy(m_terrain->nodes.data(), dev_nodes, m_terrain->nodes.size() * sizeof(Node), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(m_terrain->edges.data(), dev_edges, m_terrain->edges.size() * sizeof(Edge), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(m_terrain->modules.data(), dev_modules, m_terrain->modules.size() * sizeof(Module), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(m_terrain->moduleEdges.data(), dev_moduleEdges, m_terrain->moduleEdges.size() * sizeof(ModuleEdge), cudaMemcpyDeviceToHost));

    cudaFree(dev_modules);
    cudaFree(dev_edges);
    cudaFree(dev_nodes);

    cudaDeviceSynchronize();
}

/********************
* copyBranchesToVBO *
*********************/
__global__ void kernUpdateVBOBranches(int N, float* vbo, Node* nodes, Edge* edges)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    Edge& edge = edges[index];
    Node& fromNode = nodes[edge.fromNode];
    Node& toNode = nodes[edge.toNode];

    vbo[8 * index + 0] = fromNode.position.x;
    vbo[8 * index + 1] = fromNode.position.y;
    vbo[8 * index + 2] = fromNode.position.z;
    vbo[8 * index + 3] = fromNode.radius;

    vbo[8 * index + 4] = toNode.position.x;
    vbo[8 * index + 5] = toNode.position.y;
    vbo[8 * index + 6] = toNode.position.z;
    vbo[8 * index + 7] = toNode.radius;
}

void Simulation::copyBranchesToVBO(float* vbodptr_branches)
{
    // TODO: implement
    dim3 fullBlocksPerGrid((numOfEdges + blockSize - 1) / blockSize);
    kernUpdateVBOBranches << <fullBlocksPerGrid, blockSize >>> (numOfEdges, vbodptr_branches, dev_nodes, dev_edges);
}
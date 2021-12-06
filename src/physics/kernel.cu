#include "kernel.h"
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/copy.h>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

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

// indices of tree buffers (needed for culling)
int* dev_moduleIndices;
int* dev_temp_moduleIndices;

// Grid Dimensions
// TODO: make these dynamic to the scene being loaded
int3 gridCount = { 24, 8, 24 };        
float3 gridSize = { 60.f, 20.f, 60.f };
float sideLength = 2.5f; // "blockSize"

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

// Smoke rendering variables
float* smokeQuadsPositions;
unsigned int* smokeIndexes;
float* smokeQuadsColors;
GLuint smokeQuadVBO;
GLuint smokeQuadIndexBO;
float3 externalForce;
GLuint smokeColorBufferObj = 0;

int flatten(const int i_x, const int i_y, const int i_z) {
    return i_x + i_y * gridCount.x + i_z * gridCount.y * gridCount.z;
}

void Simulation::initSmokeQuads() {
    int nflat = gridCount.x * gridCount.y * gridCount.z;
    smokeQuadsPositions = new float[3 * 4 * 3 * nflat];
    smokeQuadsColors = new float[4 * 4 * nflat];
    smokeIndexes = new unsigned int[4 * nflat];
    for (unsigned int x = 0; x < gridCount.x; x++) {
        for (unsigned int y = 0; y < gridCount.y; y++) {
            for (unsigned int z = 0; z < gridCount.z; z++) {
                std::array<float, 12> vertexes = {
                    x * sideLength, y * sideLength, z * sideLength,
                    x * sideLength, y * sideLength, (z + 1) * sideLength,
                    x * sideLength, (y + 1) * sideLength, (z + 1) * sideLength,
                    x * sideLength, (y + 1) * sideLength, z * sideLength
                };
                std::copy(vertexes.begin(), vertexes.end(),
                    smokeQuadsPositions + 12 * flatten(x, y, z));
            }
        }
    }
    for (unsigned int x = 0; x < gridCount.x; x++) {
        for (unsigned int y = 0; y < gridCount.x; y++) {
            for (unsigned int z = 0; z < gridCount.z; z++) {
                std::array<float, 12> vertexes = {
                    x * sideLength, y * sideLength, z * sideLength,
                    (x + 1) * sideLength, y * sideLength, z * sideLength,
                    (x + 1) * sideLength, y * sideLength, (z + 1) * sideLength,
                    x * sideLength, y * sideLength, (z + 1) * sideLength
                };
                std::copy(vertexes.begin(), vertexes.end(),
                    smokeQuadsPositions + 1 * 4 * 3 * nflat + 12 * flatten(x, y, z));
            }
        }
    }
    for (unsigned int x = 0; x < gridCount.x; x++) {
        for (unsigned int y = 0; y < gridCount.y; y++) {
            for (unsigned int z = 0; z < gridCount.z; z++) {
                std::array<float, 12> vertexes = {
                    x * sideLength, y * sideLength, z * sideLength,
                    (x + 1) * sideLength, y * sideLength, z * sideLength,
                    (x + 1) * sideLength, (y + 1) * sideLength, z * sideLength,
                    x * sideLength, (y + 1) * sideLength, z * sideLength
                };
                std::copy(vertexes.begin(), vertexes.end(),
                    smokeQuadsPositions + 2 * 4 * 3 * nflat + 12 * flatten(x, y, z));
            }
        }
    }


    glGenBuffers(1, &smokeQuadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, smokeQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, 3 * nflat * 4 * 3 * sizeof(float), smokeQuadsPositions, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &smokeQuadIndexBO);
    for (int i = 0; i < 4 * nflat; i++) smokeIndexes[i] = i;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, smokeQuadIndexBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * nflat * sizeof(unsigned int), smokeIndexes, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Simulation::renderSmokeQuads(unsigned int cameraAxis) {
    int nflat = gridCount.x * gridCount.y * gridCount.z;
    glDisable(GL_CULL_FACE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glBindBuffer(GL_ARRAY_BUFFER, smokeQuadVBO);
    glVertexPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(cameraAxis * 4 * 3 * nflat * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, smokeColorBufferObj);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, smokeQuadIndexBO);
    glDrawElements(GL_QUADS, 4 * nflat, GL_UNSIGNED_INT, (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void Simulation::renderExternalForce(float3 externalForce) {
    glBegin(GL_LINES);
    glColor3f(1, 0, 0);
    glVertex3f(0, 0, 0);
    glVertex3fv((GLfloat*)&externalForce);
    glEnd();
}

void Simulation::render(unsigned int cameraAxis) {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glScalef(2, 2, 2);
    glTranslatef(-gridSize.x / 2, -gridSize.y / 2, -gridSize.z / 2);
    //if (gridEnabled) renderGrid();
    //if (raysEnabled) renderLightRays();
    renderSmokeQuads(cameraAxis);
    glPopMatrix();

    //renderExternalForce();
}

/******************
* initSimulation *
******************/

void Simulation::initSimulation(Terrain* terrain)
{
    m_terrain = terrain;
    numOfModules = terrain->modules.size();
    numOfEdges = terrain->edges.size();

    int numOfNodes = terrain->nodes.size();
    int numOfGrids = gridCount.x * gridCount.y * gridCount.z;

    // Allocate buffers for the modules
    HANDLE_ERROR(cudaMalloc((void**)&dev_nodes, numOfNodes * sizeof(Node)));
    HANDLE_ERROR(cudaMemcpy(dev_nodes, terrain->nodes.data(), numOfNodes * sizeof(Node), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_edges, numOfEdges * sizeof(Edge)));
    HANDLE_ERROR(cudaMemcpy(dev_edges, terrain->edges.data(), numOfEdges * sizeof(Edge), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_modules, numOfModules * sizeof(Module)));
    HANDLE_ERROR(cudaMemcpy(dev_modules, terrain->modules.data(), numOfModules * sizeof(Module), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_moduleEdges, terrain->moduleEdges.size() * sizeof(ModuleEdge)));
    HANDLE_ERROR(cudaMemcpy(dev_moduleEdges, terrain->moduleEdges.data(), terrain->moduleEdges.size() * sizeof(ModuleEdge), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_moduleIndices, numOfModules * sizeof(Module)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_temp_moduleIndices, numOfModules * sizeof(Module)));

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

    dim3 modules_fullBlocksPerGrid((numOfModules + blockSize - 1) / blockSize);
    
    kernInitModules << <modules_fullBlocksPerGrid, blockSize >> > (numOfModules, dev_nodes, dev_edges, dev_modules);

    kernInitIndices << <modules_fullBlocksPerGrid, blockSize >> > (numOfModules, dev_moduleIndices);

    glGenBuffers(1, &smokeColorBufferObj);
    glBindBuffer(GL_ARRAY_BUFFER, smokeColorBufferObj);

    glBufferData(GL_ARRAY_BUFFER, numOfGrids * 4 * 4 * sizeof(GLubyte), 0, GL_STREAM_DRAW);
    cudaGLRegisterBufferObject(smokeColorBufferObj);
    initSmokeQuads();

    cudaDeviceSynchronize();
}

/******************
* stepSimulation *
******************/

struct is_negative
{
    __host__ __device__
        bool operator()(int x)
    {
        return x < 0;
    }
};

struct is_nonnegative
{
    __host__ __device__
        bool operator()(int x)
    {
        return x >= 0;
    }
};

void Simulation::stepSimulation(float dt)
{
    dim3 fullBlocksPerGrid((numOfModules + blockSize - 1) / blockSize);

    // For each module in the forest
    // - Update mass
    // - Perform radii update
    // - Update temperature
    // - Update released water content
    kernModuleCombustion << <fullBlocksPerGrid, blockSize >> > (dt, numOfModules, dev_moduleIndices, gridCount, sideLength, dev_nodes, dev_edges, dev_modules, dev_moduleEdges, dev_oldtemp);

    // For each grid point x in grid space
    // - update mass and water content
    // TODO: implement
    const dim3 gridDim(blocksNeeded(gridCount.x, M_IX), blocksNeeded(gridCount.y, M_IY), blocksNeeded(gridCount.z, M_IZ));
    
    kernComputeChangeInMass<<<gridDim, M_in>>>(gridCount, numOfModules, sideLength, dev_moduleIndices, dev_modules, dev_deltaM);

    // Update air temperature
    // update drag forces (wind)
    // update smoke density (qs), water vapor (qv), condensed water (qc),
    // and rain (qc)
    uchar4* d_out = 0;
    cudaGLMapBufferObject((void**)&d_out, smokeColorBufferObj);

    float* dev_lap;
    float3* dev_alpha_m;
    float3 externalForce = { 0.f, 0.f, 0.f };

    HANDLE_ERROR(cudaMalloc(&dev_lap, gridCount.x * gridCount.y * gridCount.z * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_alpha_m, gridCount.x * gridCount.y * gridCount.z * sizeof(float3)));

    computeVorticity << <gridDim, M_in >> > (gridCount, sideLength, dev_vorticity, dev_oldvel, dev_ccvel);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());

    velocityKernel << <gridDim, M_in >> > (gridCount, gridSize, sideLength, dev_oldtemp, dev_vel, dev_oldvel, dev_alpha_m, dev_oldsmokedensity, dev_vorticity, externalForce);
    HANDLE_ERROR(cudaPeekAtLastError()); HANDLE_ERROR(cudaDeviceSynchronize());

    // Pressure Solve
    forceIncompressibility(gridCount, sideLength, dev_vel, dev_pressure);

    tempAdvectionKernel << <gridDim, M_in >> > (gridCount, gridSize, sideLength, dev_temp, dev_oldtemp, dev_vel, dev_alpha_m, dev_lap, dev_deltaM);
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

    smokeUpdateKernel << <gridDim, M_in >> > (gridCount, gridSize, sideLength, dev_oldtemp, dev_vel, dev_alpha_m, dev_smokedensity, 
        dev_oldsmokedensity, dev_deltaM);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaFree(dev_alpha_m));
    HANDLE_ERROR(cudaFree(dev_lap));

    cudaGLUnmapBufferObject(smokeColorBufferObj);

    // Ping-pong buffers
    std::swap(dev_temp, dev_oldtemp);
    std::swap(dev_vel, dev_oldvel);
    std::swap(dev_smokedensity, dev_oldsmokedensity);

    // For each module in the forest
    // cull modules (and their children) that have zero mass
    
    kernCullModules << <fullBlocksPerGrid, blockSize >> > (numOfModules, dev_moduleIndices, dev_modules, dev_edges);

    // stream compaction
    thrust::device_ptr<int> thrust_indices = 
        thrust::device_pointer_cast(dev_moduleIndices);
    thrust::device_ptr<int> thrust_temp =
        thrust::device_pointer_cast(dev_temp_moduleIndices);

    int numCulled = thrust::count_if(thrust_indices, thrust_indices + numOfModules * sizeof(int), is_negative());
    thrust::copy_if(thrust_indices, thrust_indices + numOfModules * sizeof(int), thrust_temp, is_nonnegative());

    // update data
    numOfModules -= numCulled;
    dev_moduleIndices = thrust::raw_pointer_cast(thrust_indices);
    dev_temp_moduleIndices = thrust::raw_pointer_cast(thrust_temp);

    // ping pong buffers
    std::swap(dev_moduleIndices, dev_temp_moduleIndices);
    
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

// TODO: need to cull edges
__global__ void kernUpdateVBOBranches(int N, float* vbo, Node* nodes, Edge* edges, Module* modules)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    Edge& edge = edges[index];

    if (!edge.culled)
    {
        Node& fromNode = nodes[edge.fromNode];
        Node& toNode = nodes[edge.toNode];

        vbo[11 * index + 0] = fromNode.position.x;
        vbo[11 * index + 1] = fromNode.position.y;
        vbo[11 * index + 2] = fromNode.position.z;
        vbo[11 * index + 3] = fromNode.radius;

        vbo[11 * index + 4] = toNode.position.x;
        vbo[11 * index + 5] = toNode.position.y;
        vbo[11 * index + 6] = toNode.position.z;
        vbo[11 * index + 7] = toNode.radius;

        vbo[11 * index + 8] = (fromNode.leaf && modules[edge.moduleInx].temperature < 300) ? 1.0f : -1.f;
        vbo[11 * index + 9] = (toNode.leaf && modules[edge.moduleInx].temperature < 300) ? 1.0f : -1.f;
        vbo[11 * index + 10] = modules[edge.moduleInx].temperature;
    }
    else
    {
        for (int i = 0; i < 11; i++)
        {
            vbo[11 * index + i] = 0.f;
        }
    }
}

void Simulation::copyBranchesToVBO(float* vbodptr_branches)
{
    // TODO: implement
    dim3 fullBlocksPerGrid((numOfEdges + blockSize - 1) / blockSize);
    kernUpdateVBOBranches << <fullBlocksPerGrid, blockSize >>> (numOfEdges, vbodptr_branches, dev_nodes, dev_edges, dev_modules);
    cudaDeviceSynchronize();
}
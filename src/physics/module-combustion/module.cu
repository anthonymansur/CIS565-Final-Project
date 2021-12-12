#define _USE_MATH_DEFINES // Keep above math.h import
#include <math.h> 
#include "module.h"

/************
* Constants *
*************/
/**
 * @brief the temperatures used in combustion
 * @note min: 150, max: 450, units: celsius
 */
__device__ const float T0 = 150;
__device__ const float T1 = 450;

__device__ const float T_amb = 15.f; // TODO: move elsewhere?

/**
 * @brief the saturation temperature of water
 * @note constant, units: celsius
 */
 __device__ const float T_sat = 100;

/**
 * @brief the maximum wind boost
 * @note range: XXX, units: 2 kg s^-1 m^-2
 */
__device__ const float n_max = 2;

/**
 * @brief the maximum wind velocity
 */
__device__ const float u_ref = 15;

/**
 * @brief char contraction factor
 * @note range: [0.5, 1.0], units: 1
 */
__device__ const float k = 0.75;

/**
 * @brief minimum value of c as a result of charring
 * @note range: [0.0, 1.0], units: 1
 */
__device__ const float c_min = 0.5;

/**
 * @brief the rate of insulation due to charring
 * @note range: [50,250], units: 1 m^-1
 */
__device__ const float c_r = 150;

/**
 * @brief the rate of insulation due to charring
 * @note range: constant, units: 1 Wm^-2 celsius^-1
 */
__device__ const float c_bar = 0.1;

/**
 * @brief Specific heat capacity of a module
 * @note range: constant, units: 1 kJ celsius^-1 kg
 */
__device__ const float c_M = 2.5;

/**
 * @brief density of wood
 * @note units: 1 kg m^-3, deciduous = 660, conifer = 420, shrub = 300
 */
__device__ const float rho = 500; // WARNING: see note below.
// NOTE: center of mass calculations currently assumes this to be equal to 500.

/**
 * @brief Temperature diffusion coeff. (wood)
 * @note range: XXX, units: 1 m^2 s^-1
 */
__device__ float alpha = 0.02; // TODO: first paper had different numbers?

 /**
 * @brief Temperature diffusion coeff. (module)
 * @note range: XXX, units: 1 m^2 s^-1
 */
__device__ float alpha_M = 0.75;

/** TODO: add description */
__device__ float lap_constant = 15.f;//1.648f; // TODO: TUNE

/**
 * @brief Heat transfer coeff. for dry wood 
 * @note range: [0.03, 0.1], units: 1 s^-1
 */
#define B_DRY 0.01
__device__ const float b_dry = B_DRY; // TODO: TUNE

__device__ const float rateMod = 0.1f;

/**
 * @brief Heat transfer coeff. for wet wood 
 * @note range: 0.1 * b_dry, units: 1 s^-1
 */
__device__ const float b_wet = 0.1 * B_DRY;

/**
 * @brief ratio of water released to wood burned
 * @note range: 0.5362 kg water per kg of wood
 */
__device__ const float c_WM = 0.5362;

/** TODO: add description */
__device__ const float MASS_EPSILON = FLT_EPSILON; // TODO: update
__device__ const float MAX_DELTA_M = 100;// 0.0001;  // TODO: TUNE 
__device__ const float MAX_DELTA_T = 100;//0.001; // TODO: TUNE

/*******************
* Device Functions *
********************/
__device__ float sigmoid(float x)
{
    return 3 * x * x - 2 * x * x * x;
}

__device__ float getArea(float r0, float r1, float l)
{
    return (float)M_PI * (r0 + r1) * sqrtf((r0 - r1) * (r0 - r1) + l * l);
}

__device__ float getVolume(float r0, float r1, float l)
{
    return (float)(M_PI / 3) * l * (r0 * r0 + r0 * r1 + r1 * r1);
}

__device__ float windSpeedFunction(float u)
{
    return (n_max - 1) * sigmoid(u / u_ref) + 1;
}

__device__ float computeReactionRate(float temp, float windSpeed)
{
    if (temp < T0)
        return 0;
    else if (temp > T1)
        return 1;
    else
        return windSpeedFunction(windSpeed) * sigmoid((temp - T0) / (T1 - T0));
}

__device__ float heightOfPyrolyzingFront(float H0, float A0, float mass)
{
    return sqrt(2 * (mass / rho) * (H0 / A0));
}

__device__ float charLayerThickness(float H0, float H)
{
    return k * (H0 - H);
}

__device__ float charLayerInsulation(float H_c)
{
    return c_min + (1 - c_min)*exp(-c_r * H_c);
}

__device__ float getFrontArea(float A0, float H0, float H)
{
    return A0 * H / H0;
}

__device__ float rateOfMassChange(float mass, float H0, float H, float A0, float temp, float frontArea, float windSpeed)
{
    float H_c = charLayerThickness(H0, H);
    float c = charLayerInsulation(H_c);
    float k = computeReactionRate(temp, windSpeed);

    // TODO: verify this is correct, as it's throwing nan
    return -1 * k * c * frontArea * rateMod;
}

__device__ float radiiModuleConstant(Node* nodes, Edge* edges, Module& module)
{
    /** Replace code with what's commented for simplier solution */
    //Node& node = nodes[module.startNode];
    //return node.radius / sqrt((3 / (M_PI * rho)) * module.mass);

    float sum = 0;
    for (int i = module.startEdge; i <= module.lastEdge; i++)
    {
        // For every edge in the module, do the following:

        Edge* edge = &edges[i]; // will be updated
        float l = edge->length;
        float lambda = edge->radiiRatio;
        float prod = 1;

        // check to see if fromNode isn't the root node
        while (edge->fromNode != module.startNode)
        {
            // Need to traverse every edge in the path from root node to 
            // the initial edge's fromNode. To do so, we will do the following
            
            // go to the current node's previous edge
            int nodeInx = edge->fromNode;
            edge = &edges[nodes[nodeInx].previousEdge];

            // compute the product
            float _lambda = edge->radiiRatio;
            prod *= (_lambda * _lambda);
        }
        sum += l * prod * (1 + lambda + lambda * lambda);
    }
    return 1 / sqrt(sum);
}

__device__ float radiiUpdateRootNode(Node* nodes, Edge* edges, Module& module, float deltaMass)
{
    if (module.mass + deltaMass < FLT_EPSILON) return 0.f;
    return sqrt(3 / (M_PI * rho)) * radiiModuleConstant(nodes, edges, module) * sqrt(module.mass + deltaMass);
}

// TODO: verify this is correct before adding it to kernel! 
__device__ float radiiUpdateNode(Node* nodes, Edge* edges, Module& module, int nodeInx, float rootRadius)
{

    if (rootRadius < FLT_EPSILON) return 0.f;
    int currNodeInx = nodeInx;
    Edge* edge; // will be updated
    float prod = 1;
    do
    {
        // Need to traverse every edge in the path from root node to 
        // the node given. To do so, we will do the following

        // go to the current node's previous edge
        Node& node = nodes[currNodeInx];
        edge = &edges[node.previousEdge];
        currNodeInx = edge->fromNode;

        // compute the product
        prod *= edge->radiiRatio;
    } while (currNodeInx != module.startNode);
    return prod * rootRadius;
}

// TODO: diffusion of adjacent modules not yet correctlyimplemented
__device__ float rateOfTemperatureChange(float T, float T_M, float T_diff, float W, float A_M, float V_M)
{
    float b = (1 - W) * b_dry + W * b_wet;
    float diffusion = alpha_M * T_diff; // TODO: implement diffusion. see eq. (30)
    float temp_diff = b * (T - T_M);
    // TODO: add back change of energy

    float changeOfEnergy = 0;
    if (T_M > 150) // start of combustion
        changeOfEnergy = (c_bar * A_M * powf(T_M - T_sat, 3)) / (V_M * rho * c_M);

    return diffusion + temp_diff - changeOfEnergy; 
}

__device__ float rateOfWaterChange(float changeInMass)
{
    return c_WM * changeInMass;
}

// TODO: transfer these function calls to the fluid solver
__device__ float getDeltaMassOfModuleAtPoint(Module& module, glm::vec3 x, float dx)
{
    return (1 - glm::distance(x, module.centerOfMass) / dx) * module.deltaM;
}

__device__ float getWaterOfModuleAtPoint(Module& module, glm::vec3 x, float dx)
{
    return (1 - glm::distance(x, module.centerOfMass) / dx) * module.waterContent;
}

__device__ float checkModuleIntersection(Module& module, glm::vec3 pos)
{
    bool intersects = true;
    for (int i = 0; i < 3; i++)
    {
        if (pos[i] < module.boundingMin[i] || pos[i] > module.boundingMax[i])
        {
            intersects = false;
            break;
        }
    }
    return intersects;
}

// TODO: add prototype to header file
__device__ float getModuleTemperatureLaplacian(Module* modules, ModuleEdge* moduleEdges, int moduleInx)
{
    Module& module = modules[moduleInx];
    float lap = 0.f;

    if (module.startModule < 0 || module.endModule < 0)
        return 0.f; 

    for (int i = module.startModule; i <= module.endModule; i++)
    {
        Module& adj = modules[moduleEdges[i].moduleInx];
        if (adj.culled) continue;
        float dist = glm::distance(module.centerOfMass, adj.centerOfMass);
        lap += (adj.temperature - module.temperature) / (dist * dist);
    }
    return lap * lap_constant;
}

/**********
* Kernels *
***********/

__global__ void kernInitModules(int N, Node* nodes, Edge* edges, Module* modules)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    Module& module = modules[index];

    if (module.startEdge < 0 || module.lastEdge < 0)
        return;

    float area = 0.f;
    float mass = 0.f;
    for (int i = module.startEdge; i <= module.lastEdge; i++)
    {
        // for every edge in the module
        Edge& edge = edges[i];
        Node& fromNode = nodes[edge.fromNode];
        float r0 = fromNode.radius;
        float r1 = edge.radiiRatio * r0;
        float l = edge.length;
        area += getArea(r0, r1, l);
        float volume = getVolume(r0, r1, l);
        mass += volume * rho; // mass = density * volume 
    }

    glm::vec3 minPos{FLT_MAX}, maxPos{FLT_MIN};
    for (int i = module.startNode; i <= module.lastNode; i++)
    {
        // for every node in the module
        Node& node = nodes[i];
        glm::vec3 pos = node.position;
        for (int j = 0; j < 3; j++)
        {
            if (pos[j] < minPos[j])
                minPos[j] = pos[j];
            if (pos[j] > maxPos[j])
                maxPos[j] = pos[j];
        }
    }

    module.mass = mass;
    module.startArea = area;
    module.moduleConstant = radiiModuleConstant(nodes, edges, module);
    module.boundingMin = minPos;
    module.boundingMax = maxPos;

    module.deltaM = 0.f;
    module.temperature = T_amb;

    // primitive combustion
    /*if (index < 4000)
        module.temperature = 300;*/

    module.waterContent = 0.f;
}

__global__ void kernInitIndices(int N, int* indices)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    indices[index] = index;
}

__global__ void kernModuleCombustion(float DT, int N, int* moduleIndices, int3 gridCount, float blockSize, Node* nodes, Edge* edges, Module* modules, ModuleEdge* moduleEdges, float* gridTemp)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    int moduleIndex = moduleIndices[index];

    if (moduleIndex == -1) return;

    /** for each module in the forest */
    Module& module = modules[moduleIndex];
    Node& rootNode = nodes[module.startNode];

    float oldMass = module.mass;

    // Module needs to be culled
    if (module.mass < MASS_EPSILON || module.startEdge < 0 || module.lastEdge < 0 || module.culled) return;

    /** 1. update the mass */
    // calculate the current state of the module
    float mass = 0.f;
    float area = 0;
    float temp = module.temperature; 

    for (int i = module.startEdge; i <= module.lastEdge; i++)
    {
        // for every branch in the module
        Edge& edge = edges[i];
        Node& fromNode = nodes[edge.fromNode];

        float r0 = fromNode.radius;

        float r1 = edge.radiiRatio * r0;
        float l = edge.length;

        float volume = getVolume(r0, r1, l);
        mass += volume * rho; // mass = density * volume 
        area += getArea(r0, r1, l);
    }

    // compute the change in mass
    float H0 = rootNode.startRadius;
    float A0 = module.startArea;
    float H = heightOfPyrolyzingFront(H0, A0, mass);
    float frontArea = getFrontArea(A0, H0, H);
    float windSpeed = 0; // TODO: implement
    //float deltaM = glm::clamp(rateOfMassChange(mass, H0, H, A0, temp, frontArea, windSpeed), -MAX_DELTA_M, 0.f);
    float deltaM = rateOfMassChange(mass, H0, H, A0, temp, frontArea, windSpeed);
    //if (deltaM != deltaM) deltaM = -0.001f;
    /*if (moduleIndex < 12500 && module.temperature > 150)
        deltaM = -0.0005f;
    else
        deltaM = 0.f;*/

    module.mass += deltaM;
    module.deltaM = deltaM;

    /** Perform radii update */
    // update the root's radius
    float rootRadius = radiiUpdateRootNode(nodes, edges, module, deltaM);
    rootNode.radius = rootRadius;

    for (int i = module.startNode + 1; i <= module.lastNode; i++)
    {
        // update the radius of each branch in the module
        Node& node = nodes[i];
        float newRadius = radiiUpdateNode(nodes, edges, module, i, rootRadius);
        node.radius = newRadius;
    }

    // TODO: sync threads here?

    /** 3. Update temperature */
    float T_env = gridTemp[module.gridCell];
    float T_diff = getModuleTemperatureLaplacian(modules, moduleEdges, moduleIndex);
    float T_M = module.temperature;
    float W = 0; // TODO: get the water content
    float A_M = area; // lateral surface area 
    float V_M = module.mass / rho;
    float deltaT = glm::clamp(rateOfTemperatureChange(T_env, T_M, T_diff, W, A_M, V_M), 0.f, MAX_DELTA_T);

    module.temperature += deltaT;

    /** 4. Update released water content */
    float deltaW = rateOfWaterChange(deltaM);
    module.waterContent += deltaW;
}

// TODO: these functions are duplicated in advection.h/.cpp
__device__ int m_idxClip(int idx, int idxMax) {
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}
__device__ int m_flatten(int col, int row, int z, int width, int height, int depth) {
    return m_idxClip(col, width) + m_idxClip(row, height) * width + m_idxClip(z, depth) * width * height;
}
__global__ void kernComputeChangeInMass(int3 gridCount, Module* modules, GridCell* gridCells, GridModuleAdj* gridModuleAdjs, float* gridOfMass)
{
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x) || (k_y >= gridCount.y) || (k_z >= gridCount.z)) return;
    const int k = m_flatten(k_x, k_y, k_z, gridCount.x, gridCount.y, gridCount.z);

    GridCell& gridCell = gridCells[k];
    if (gridCell.startModule < 0 || gridCell.endModule < 0)
        return;

    float deltaM = 0.f;
    for (int i = gridModuleAdjs[gridCell.startModule].moduleInx; i <= gridModuleAdjs[gridCell.endModule].moduleInx; i++)
    {
        if (modules[i].deltaM < -MAX_DELTA_M || modules[i].deltaM > 0.f) {
            continue;
        }
        deltaM += modules[i].deltaM;
    }
    gridOfMass[k] = deltaM;
}

__device__ float getGridCell(Module& module, int3 gridCount, float blockSize)
{
    // Convert center of mass to grid-space coordinates // e.g. (-10,10) to (0, 20)
    glm::vec3 com = module.centerOfMass;
    com.x += floor(gridCount.x * blockSize / 2);
    com.y += floor(gridCount.y * blockSize / 2);
    com.z += floor(gridCount.z * blockSize / 2);

    // get the grid at this location
    for (int i = 0; i < 3; i++)
        com[i] = blockSize * round(com[i] / blockSize);
    int inx = m_flatten(com.x, com.y, com.z, gridCount.x, gridCount.y, gridCount.z);

    return inx;
}

__device__ float getEnvironmentTempAtModule(Module& module, float* temp, int3 gridCount, float blockSize)
{
    int inx = getGridCell(module, gridCount, blockSize);

    return temp[inx];
}

// TODO: cull children of modules
__global__ void kernCullModules1(int N, int* moduleIndices, Module* modules, ModuleEdge* moduleEdges, Node* nodes, Edge* edges)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    Module& module = modules[moduleIndices[index]];

    // check if module needs to be culled
    if (module.mass < MASS_EPSILON || module.startEdge < 0 || module.lastEdge < 0)
    {
        module.culled = true;

        // cull children
        for (int i = module.startModule; i < module.endModule; i++)
        {
            modules[moduleEdges[i].moduleInx].culled = true;
        }
    }
}

__global__ void kernCullModules2(int N, int* moduleIndices, Module* modules, ModuleEdge* moduleEdges, Node* nodes, Edge* edges)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    Module& module = modules[moduleIndices[index]];

    // check if module needs to be culled
    if (module.culled)
    {
        moduleIndices[index] = -1; // cull the module

        if (module.startEdge < 0 || module.lastEdge < 0)
            return;

        // for every edge in the module, cull it so it isn't rendered
        for (int i = module.startEdge; i <= module.lastEdge && i >= 0; i++)
        {
            Edge& edge = edges[i];
            edge.culled = true;
            nodes[edge.fromNode].radius = 0.f;
            nodes[edge.toNode].radius = 0.f;
        }
    }
}
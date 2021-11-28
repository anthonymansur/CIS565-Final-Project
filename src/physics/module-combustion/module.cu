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
__device__ const float T0 = 150, T1 = 450;

__device__ const float T_amb = 15; // TODO: move elsewhere?

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
__device__ const float rho = 500;

/**
 * @brief Temperature diffusion coeff. (wood)
 * @note range: XXX, units: 1 m^2 s^-1
 */
 float alpha = 0.02; // TODO: first paper had different numbers?

 /**
 * @brief Temperature diffusion coeff. (module)
 * @note range: XXX, units: 1 m^2 s^-1
 */
 float alpha_M = 0.75;

/**
 * @brief Heat transfer coeff. for dry wood 
 * @note range: [0.03, 0.1], units: 1 s^-1
 */
#define B_DRY 0.06
__device__ const float b_dry = B_DRY; 

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

/*******************
* Device Functions *
********************/
__device__ float sigmoid(float x)
{
    3 * x * x - 2 * x * x * x;
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

__device__ float rateOfMassChange(float mass, float H0, float A0, float temp, float frontArea, float windSpeed)
{
    float H = heightOfPyrolyzingFront(H0, A0, mass);
    float H_c = charLayerThickness(H0, H);
    float c = charLayerInsulation(H_c);
    float k = computeReactionRate(temp, windSpeed);
    return -1 * k * c * frontArea;
}

// TODO: verify this is correct before adding it to kernel! 
__device__ float radiiModuleConstant(Node* nodes, Edge* edges, Module& module)
{
    float moduleConstant = sqrt(3 / M_PI * rho);
    float sum = 0;
    for (int i = module.startEdge; i <= module.lastEdge; i++)
    {
        // For every edge in the module, do the following:

        Edge* edge = &edges[i]; // will be updated
        float l = edge->length;
        float prod = 1;
        do
        {
            // For every edge in the path of the module's root node to the 
            // edge's from node, traversing first from the edge's from node
            // to the root node, do the following: 

            float lambda = edge->radiiRatio;
            prod *= (lambda * lambda) * (1 + lambda + lambda * lambda);

            // get previous edge if not at the first edge
            if (edge->fromNode != module.startNode)
                edge = &edges[nodes[edge->fromNode].previousEdge];
        } while (edge->fromNode != module.startNode);     

        sum += l * prod;
    }

    return 1 / sqrt(sum);
}

__device__ float radiiUpdateRootNode(Node* nodes, Edge* edges, Module& module, float deltaMass)
{
    return sqrt(3 / (M_PI * rho)) * radiiModuleConstant(nodes, edges, module) * sqrt(module.mass + deltaMass);
}

// TODO: verify this is correct before adding it to kernel! 
__device__ float radiiUpdateNode(Node* nodes, Edge* edges, Module& module, int nodeInx, float rootRadius)
{
    int currNodeInx = nodeInx;
    Edge* edge; // will be updated
    float prod = 1;
    do
    {
        // For every edge in the path of the module's root node to the 
        // edge's from node, traversing first from the edge's from node
        // to the root node, do the following: 

        // TODO: use previous edge instead find the parent 
        edge = &edges[nodes[currNodeInx].previousEdge]; 
        prod *= edge->radiiRatio * rootRadius; 

        // update currNode
        currNodeInx = edge->fromNode;
    } while (edge->fromNode != module.startNode); 

    return prod;
}

// TODO: diffusion of adjacent modules not yet implemented
__device__ float rateOfTemperatureChange(float T, float T_M, float W, float A_M, float V_M)
{
    float b = (1 - W) * b_dry + W * b_wet;

    float diffusion = 0; // TODO: implement diffusion. see eq. (30)
    float temp_diff = b * (T - T_M);
    // TODO: add back change of energy
    float changeOfEnergy = 0 /*(c_bar * A_M * powf((T_M - T_sat), 3)) / (V_M * rho * c_M)*/;

    return diffusion + temp_diff - changeOfEnergy; 
}

__device__ float rateOfWaterChange(float changeInMass)
{
    return c_WM * changeInMass;
}

// NOTE: this is a very basic implementation. May need to be updated.
__device__ glm::vec3 centerOfMass(Module& module)
{
    glm::vec3 center;
    for (int i = 0; i < 3; i++)
        center[i] = (module.boundingMax[i] + module.boundingMin[i]) * 0.5f;
    return center;
}

// TODO: transfer these function calls to the fluid solver
__device__ float getMassOfModuleAtPoint(Module& module, glm::vec3 x, float dx)
{
    glm::vec3 center = centerOfMass(module);
    return (1 - glm::distance(x, center) / dx) * module.mass;
}

__device__ float getWaterOfModuleAtPoint(Module& module, glm::vec3 x, float dx)
{
    glm::vec3 center = centerOfMass(module);
    return (1 - glm::distance(x, center) / dx) * module.waterContent;
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

/**********
* Kernels *
***********/

__global__ void kernInitModules(int N, Node* nodes, Edge* edges, Module* modules)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    Module& module = modules[index];
    float area;
    float mass;
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

    module.deltaM = 0;
    if (index == 1)
        module.temperature = (T0 + T1) / 2; // set module 1 on fire
    else
        module.temperature = T_amb;
    module.waterContent = 0;
}

__global__ void kernModuleCombustion(float DT, int N, int3 gridCount, float blockSize, Node* nodes, Edge* edges, Module* modules, float* gridTemp)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    /** for each module in the forest */
    Module& module = modules[index];
    Node& rootNode = nodes[modules->startNode];

    /** 1. update the mass */
    // calculate the current state of the module
    float mass = 0;
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
    float H0 = rootNode.radius;
    float A0 = module.startArea;
    float H = heightOfPyrolyzingFront(H0, A0, mass);
    float frontArea = getFrontArea(A0, H0, H);
    float windSpeed = 0; // TODO: implement
    float deltaM = rateOfMassChange(mass, H0, A0, temp, frontArea, windSpeed);
    
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
    float T = getAverageValue(gridTemp, gridCount, blockSize, module.boundingMin, module.boundingMax);
  
    float T_M = module.temperature;
    float W = 0; // TODO: get the water content
    float A_M = area; 
    float V_M = module.mass / rho;
    float deltaT = rateOfTemperatureChange(T, T_M, W, A_M, V_M);

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
__global__ void kernComputeChangeInMass(int3 gridCount, int numOfModules, float blockSize, Module* modules, float* gridOfMass)
{
    const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int k_z = threadIdx.z + blockDim.z * blockIdx.z;
    if ((k_x >= gridCount.x) || (k_y >= gridCount.y) || (k_z >= gridCount.z)) return;
    const int k = m_flatten(k_x, k_y, k_z, gridCount.x, gridCount.y, gridCount.z);

    float3 pos = make_float3((k_x + 0.5f) * blockSize, (k_y + 0.5f) * blockSize, (k_z + 0.5f) * blockSize);
    // converting grid-space to terrain-space
    glm::vec3 glmPos{ 0.f };
    glmPos.x = pos.x - floor(gridCount.x * blockSize / 2);
    glmPos.y = pos.y - floor(gridCount.y * blockSize / 2);
    glmPos.z = pos.z - floor(gridCount.z * blockSize / 2);

    float deltaM = 0;
    for (int i = 0; i < numOfModules; i++)
    {
        if (checkModuleIntersection(modules[i], glmPos))
            deltaM += getMassOfModuleAtPoint(modules[i], glm::vec3(pos.x, pos.y, pos.z), blockSize);
    }
    gridOfMass[k] = deltaM;
}

// TODO: may need to change implementation
__device__ float getAverageValue(float* buffer, int3 gridCount, float blockSize, glm::vec3 min, glm::vec3 max)
{
    // convert terrain-space to grid-space coordinates // e.g. (-10,10) to (0, 20)
    min.x += floor(gridCount.x * blockSize / 2);
    min.y += floor(gridCount.y * blockSize / 2);
    min.z += floor(gridCount.z * blockSize / 2);
    max.x += floor(gridCount.x * blockSize / 2);
    max.y += floor(gridCount.y * blockSize / 2);
    max.z += floor(gridCount.z * blockSize / 2);

    glm::vec3 minPoint, maxPoint;
    for (int i = 0; i < 2; i++)
    {
        // get the min and max coord
        minPoint[i] = blockSize * round(min[i] / blockSize); // TODO: is there a better way to do this?
        maxPoint[i] = blockSize * round(max[i] / blockSize);
    }

    // iterate over all the grid cells
    float sum = 0.f;
    int num = 0;
    for (float i = minPoint.x; i <= maxPoint.x; i += blockSize)
    {
        for (float j = minPoint.y; j <= maxPoint.y; j += blockSize)
        {
            for (float k = minPoint.z; k <= maxPoint.z; k += blockSize)
            {
                if ((i >= gridCount.x) || (j >= gridCount.y) || (k >= gridCount.z)) continue;
                int inx = m_flatten(i, j, k, gridCount.x, gridCount.y, gridCount.z);
                sum += buffer[inx];
                num++;
            }
        }
    }

    return num > 0 ? (sum / num) : 0;
}
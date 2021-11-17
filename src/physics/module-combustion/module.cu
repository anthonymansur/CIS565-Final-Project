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
 * @brief density of wood
 * @note units: 1 kg m^-3, deciduous = 660, conifer = 420, shrub = 300
 */
__device__ const float rho = 500;

/**
 * @brief Temperature diffusion coeff. (wood)
 * @note range: [0.1, 1.0] x 10^-7, units: 1 m^2 s^-1
 */
 float a = 0.5 * powf(10, -7);

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

/***************************
* Function implementations *
****************************/
__device__ float sigmoid(float x)
{
    3 * x * x - 2 * x * x * x;
}

__device__ float area(float r0, float r1, float l)
{
    return (float)M_PI * (r0 + r1) * sqrtf((r0 - r1) * (r0 - r1) + l * l);
}

__device__ float volume(float r0, float r1, float l)
{
    return (float)(M_PI / 3) * l * (r0 * r0 + r0 * r1 + r1 * r1);
}

__device__ float windSpeedFunction(float u)
{
    (n_max - 1)* sigmoid(u / u_ref) + 1;
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

__device__ float frontArea(float A0, float H0, float H)
{
    return A0 * H / H0;
}

__device__ float rateOfMassChange(float mass, float H0, float A0, float temp, float frontArea, float windSpeed)
{
    float H = heightOfPyrolyzingFront(H0, A0, mass);
    float H_c = charLayerThickness(H0, H);
    float c = charLayerInsulation(H_c);
    return -1 * computeReactionRate(temp, windSpeed) * c * frontArea;
}

// TODO: verify this is correct before adding it to kernel! 
__device__ float radiiModuleConstant(Node* nodes, Edge* edges, Module& module)
{
    float moduleConstant = sqrt(3 / M_PI * rho);
    float sum;
    for (int i = module.startEdge; i <= module.lastEdge; i++)
    {
        // For every edge in the module, do the following:

        Edge* edge = &edges[i]; // will be updated
        float l = edge->length;
        float prod;
        int pathEdge = -1;
        do
        {
            // For every edge in the path of the module's root node to the 
            // edge's from node, traversing first from the edge's from node
            // to the root node, do the following: 

            // find the parent 
            for (int i = module.startEdge; i <= module.lastEdge; i++)
            {
                if (edges[i].toNode == edge->fromNode)
                {
                    // if we found the to edge that matches this node, we can then
                    // find the parent of the node.
                    pathEdge = i;
                    break;
                }
            }
            if (pathEdge < 0)
                return -1; // error in the graph construction

            float lambda = edge->radiiRatio;

            prod *= (lambda * lambda) * (1 + lambda + lambda * lambda);

            // get previous edge if not at the first edge
            if (edge->fromNode != module.startEdge)
                edge = &edges[pathEdge];
        } while (edge->fromNode != module.startNode);     

        sum += l * prod;
    }

    return 1 / sqrt(sum);
}

__device__ float radiiUpdateRootNode(Node* nodes, Edge* edges, Module& module, float deltaMass)
{
    return sqrt(3 / (M_PI * rho)) * radiiModuleConstant(nodes, edges, module) * sqrt(module.mass + deltaMass);
}

__device__ float radiiUpdateNode(Node* nodes, Edge* edges, Module& module, int nodeInx, float rootRadius)
{
    int currNodeInx = nodeInx;
    int pathEdge = -1;
    Edge* edge; // will be updated
    float prod;
    do
    {
        // For every edge in the path of the module's root node to the 
        // edge's from node, traversing first from the edge's from node
        // to the root node, do the following: 

        // find the parent 
        for (int i = module.startEdge; i <= module.lastEdge; i++)
        {
            if (edges[i].toNode == currNodeInx)
            {
                // if we found the to edge that matches this node, we can then
                // find the parent of the node.
                pathEdge = i;
                break;
            }
        }
        if (pathEdge < 0)
            return -1; // error in the graph construction

        edge = &edges[pathEdge]; 
        prod *= edge->radiiRatio * rootRadius;

        // get previous edge if not at the first edge
        if (edge->fromNode != module.startEdge)
            edge = &edges[pathEdge];
    } while (edge->fromNode != module.startNode); 

    return prod;
}

__device__ float rateOfTemperatureChange(float T, float T_M, float W)
{
    float b = (1 - W) * b_dry + W * b_wet;

    // TODO: implement
    // Need the laplacian of the module temperature 
    return -1; 
}

__device__ float rateOfWaterChange(float changeInMass)
{
    return c_WM * changeInMass;
}
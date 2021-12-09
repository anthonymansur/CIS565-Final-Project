#pragma once

#include "../../TerrainStructs.h"

// device function prototypes
/**
 * @brief sigmoid-like function that interpolates smoothly from zero to one for
 * temperatures between 150 and 450 degrees celsius
 * 
 * @param x the input to the sigmoid
 * 
 * @return float between 0 or 1
 */
__device__ float sigmoid(float x);

/**
 * @brief the area of the branch
 * 
 * @param r0 the starting radius of the branch
 * @param r1 the ending radius of the branch
 * @param l the length of the branch
 *
 * @return the area
 */
__device__ float getArea(float r0, float r1, float l);

/**
 * @brief the volume of the branch
 * 
 * @param r0 the starting radius of the branch
 * @param r1 the ending radius of the branch
 * @param l the length of the branch
 *
 * @return the volume
 */
__device__ float getVolume(float r0, float r1, float l);

/**
 * @brief wind function used to scale the reaction rate
 * 
 * @param u the wind speed
 *
 * @return the number to scale the reaction rate by
 */
__device__ float windSpeedFunction(float u);

/**
 * @brief computes the reaction rate used for calculate the rate of change
 * 
 * @param temp the temperature of the module 
 * @param windSpeed the wind speed
 *
 * @return the reaction rate, K
 */
__device__ float computeReactionRate(float temp, float windSpeed);

/**
 * @brief The current height of the pyrolyzing front
 * 
 * @param H0 the original depth of the volumetric portion represenbed by a 
 * surface element
 * @param A0 the starting surface area of the pyrolyzing front
 * @param mass the mass of the module
 * 
 * @return the height, H
 */
__device__ float heightOfPyrolyzingFront(float H0, float A0, float mass);

/**
 * @brief The char layer thickness of the module used in the change of mass function
 * 
 * @param H0 the original depth of the volumetric portion represenbed by a 
 * surface element
 * @param H the height of the pyrolzing front
 * 
 * @return the char layer thickness, H_c
 */
__device__ float charLayerThickness(float H0, float H);

/**
 * @brief The insulation of the module provided by the char insulation
 * 
 * @param H_c the char layer thickness
 * 
 * @return the char layer insulation, c
 */
__device__ float charLayerInsulation(float H_c);

/**
 * @brief The area of the pyrolyzing front
 * 
 * @param A0 the starting surface area of the pyrolyzing front
 * @param H0 the original depth of the volumetric portion represenbed by a 
 * surface element
 * @param H the height of the pyrolzing front
 * 
 * @return the area, A
 */
__device__ float getFrontArea(float A0, float H0, float H);

/**
 * @brief The rate of mass chass of a given module
 * 
 * @param mass the current mass of the module
 * @param H0 the original depth of the volumetric portion represented by a 
 * surface element
 * @param A0 the starting surface area of the pyrolyzing front
 * @param temp the temperature of the module
 * @param frontArea the surface area of the pyrolyzing front
 * @param windSpeed the speed of the wind
 * 
 * @return the rate
 */
__device__ float rateOfMassChange(float mass, float H0, float A0, float temp, float frontArea, float windSpeed);

/**
 * @brief Derives a constant that will be used during the radii update algorithm
 * 
 * @param nodes device pointer to all the nodes in the forest
 * @param edges device pointer to all the edges in the forest
 * @param module the current module we are updating the radii for
 * 
 * @return the constant, psi_M
 */
__device__ float radiiModuleConstant(Node* nodes, Edge* edges, Module& module);

/**
 * @brief Gets the new radius of the root branch for given module 
 * 
 * @param nodes device pointer to all the nodes in the forest
 * @param edges device pointer to all the edges in the forest
 * @param module the current module we are updating the radii for
 * @param deltaMass the change in mass of the module
 * 
 * @return the new radius
 */
__device__ float radiiUpdateRootNode(Node* nodes, Edge* edges, Module& module, float deltaMass);

/**
 * @brief Gets the new radius of the given branch in the module 
 * 
 * @param nodes device pointer to all the nodes in the forest
 * @param edges device pointer to all the edges in the forest
 * @param module the current module we are updating the radii for
 * @param nodeInx the index of the node we are updating the radius for
 * 
 * @return the new radius
 */
__device__ float radiiUpdateNode(Node* nodes, Edge* edges, Module& module, int nodeInx, float rootRadius);

/**
 * @brief Gets the rate of change of temperature in the module
 * 
 * @param T the temperature of the surrounding air
 * @param T_M the module's temperature
 * @param T_adj average temperature of adjacent modules
 * @param W the water content
 * @param A_M the surface area of the module
 * @param V_M the volume of the module
 * 
 * @return the new radius
 */
__device__ float rateOfTemperatureChange(float T, float T_M, float T_adj, float W, float A_M, float V_M);

/**
 * @brief Gets the rate of water change for a module given its change in mass
 * 
 * @param changeInMass change in mass of the module 
 * 
 * @return the rate
 */
__device__ float rateOfWaterChange(float changeInMass);

// TODO: transfer these function calls to the fluid solver
/**
 * @brief Get the change in mass of a module at a given point in space, assuming this point 
 * intersects with the module's bounding box.
 * 
 * @param module the module
 * @param x the point in space
 * @param dx delta x
 *
 * @return the mass
 */
__device__ float getDeltaMassOfModuleAtPoint(Module& module, glm::vec3 x, float dx);

/**
 * @brief Get the water content of a module at a given point in space, assuming this point 
 * intersects with the module's bounding box.
 * 
 * @param module the module
 * @param x the point in space
 * @param dx delta x
 *
 * @return the water content
 */
__device__ float getWaterOfModuleAtPoint(Module& module, glm::vec3 x, float dx);

/**
 * @brief Check if module intersects this space
 * 
 * @param module
 * @param pos
 *
 * @return true if intersects, false otherwise
 */
__device__ float checkModuleIntersection(Module& module, glm::vec3 pos);

/** TODO: add description */
__global__ void kernInitIndices(int N, int* indices);

/**
 * @brief Initializes the state of the modules before running the simulation
 * 
 * @param N the number of modules
 * @param nodes device pointer to the nodes
 * @param edges device pointer to the edges
 * @param modules device pointer to the modules
 */
__global__ void kernInitModules(int N, Node* nodes, Edge* edges, Module* modules);

/**
 * @brief Runs the module combustion algorithm
 * 
 * @param DT delta time
 * @param N the number of modules
 * @param nodes device pointer to the nodes
 * @param edges device pointer to the edges
 * @param modules device pointer to the modules
 */
// TODO: update params
__global__ void kernModuleCombustion(float DT, int N, int* moduleIndices, int3 gridCount, float blockSize, Node* nodes, Edge* edges, Module* modules, ModuleEdge* moduleEdges, float* gridTemp);

/** TODO: add description */
__global__ void kernComputeChangeInMass(int3 gridCount, Module* modules, GridCell* gridCells, GridModuleAdj* gridModuleAdjs, float* gridOfMass);

/** TODO: add description */
__device__ float getEnvironmentTempAtModule(Module& module, float* temp, int3 gridCount, float blockSize);

/** TODO: add description */
__global__ void kernCullModules(int N, int* moduleIndices, Module* modules, Edge* edges);

/** TODO: add description */
__device__ float getGridCell(Module& module, int3 gridCount, float blockSize);


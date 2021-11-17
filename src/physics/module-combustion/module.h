#pragma once

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
__device__ float area(float r0, float r1, float l);

/**
 * @brief the volume of the branch
 * 
 * @param r0 the starting radius of the branch
 * @param r1 the ending radius of the branch
 * @param l the length of the branch
 *
 * @return the volume
 */
__device__ float volume(float r0, float r1, float l);

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
__device__ float frontArea(float A0, float H0, float H);

/**
 * @brief The rate of mass chass of a given module
 * 
 * @param mass the current mass of the module
 * @param H0 the original depth of the volumetric portion represenbed by a 
 * surface element
 * @param A0 the starting surface area of the pyrolyzing front
 * @param temp the temperature of the module
 * @param frontArea the surface area of the pyrolyzing front
 * @param windSpeed the speed of the wind
 * 
 * @return the rate
 */
__device__ float rateOfMassChange(float mass, float H0, float A0, float temp, float frontArea, float windSpeed);
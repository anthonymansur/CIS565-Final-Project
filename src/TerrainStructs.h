#pragma once

#include <glm/glm.hpp>

/**
 * See: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.4206&rep=rep1&type=pdf
 * Graphs will be representing in the GPU in a compact adjacency list form.
 * with adjacency lists for every vertex arranged in a single array. A vertex
 * array will also be stored with pointers to the starting edges in the 
 * adjacency list array.
 */

struct Node
{
    // Node-specific parameters
    float radius; // starting radius of branch
    glm::vec3 position; // the location of the node in world
    
    // Pointers
    int previousEdge;
    int firstEdge; 
    int lastEdge;
};

struct Edge
{
    // Edge-specific parameters
    float length;
    float radiiRatio;
    glm::vec3 direction;

    // Pointers
    int fromNode;
    int toNode;
};

struct Module
{
    // Module-specific parameters
    float temperature;
    float mass, deltaM; 
    glm::vec3 boundingMin, boundingMax; // Bounding box for the module 
    float startArea; // lateral surface area before combustion
    float moduleConstant;
    float waterContent;

    // Pointers
    // -- modules
    int startModule;
    int endModule;
    // -- nodes
    int previousNode; // terminal node of previous module
    int startNode; // root node of module
    int lastNode; // the last node in the module's graph
    // -- edges
    int startEdge; 
    int lastEdge; // may be a connection node
    // TODO: revisit this assumption about the pointer to last node as a module has many terminal nodes! 
};

struct ModuleEdge
{
    int moduleInx;
};
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
    float startRadius; // radius before burning
    glm::vec3 position; // the location of the node in grid space
    int previousEdge;
    int firstEdge; 
    int lastEdge;
    bool leaf;
};

struct Edge
{
    // Edge-specific parameters
    float length;
    float radiiRatio;
    int moduleInx;
    bool culled = false;
    int fromNode;
    int toNode;
};

struct Module
{
    int previousNode; // terminal node of previous module

    int startNode; // root node of module
    int lastNode; // the last node in the module's graph
    int startEdge; 
    int lastEdge; // may be a connection node
    int startModule; // module edge pointer
    int endModule; // module edge pointer
    int parentModule; // pointer to parent module
    int gridCell;
    //int treeId; // id of the tree this module is associated with

    // module-level parameters
    float temperature;
    float mass, deltaM; 
    glm::vec3 boundingMin, boundingMax; // Bounding box for the module 
    glm::vec3 centerOfMass;
    float startArea; // lateral surface area before combustion
    float moduleConstant;
    //float waterContent;

    bool culled = false;
};

struct ModuleEdge
{
    int moduleInx;
};

struct GridCell
{
    int startModule;
    int endModule;
};

struct GridModuleAdj
{
    int moduleInx;
};
#include "Tree.h"
#include <stdlib.h>
#include <time.h>

float rnum(float scale)
{
    return 2 * scale * ((rand() % 2) - 0.5); // random num between -scale and scale 
} 

Tree::Tree()
{
    /* initialize random seed: */
    srand(time(NULL));
}

Tree::~Tree()
{}

#define RADIUS_REDUCTION 0.9

void addNode(Terrain* terrain, float length, glm::vec3 dir, int childInx, int parentInx)
{
    terrain->nodes[childInx].position = terrain->nodes[parentInx].position + (length * dir);
    terrain->nodes[childInx].radius = terrain->nodes[parentInx].radius * (RADIUS_REDUCTION / length);
}

void addEdge(Terrain* terrain, int edge, int fromNode, int toNode)
{
    terrain->edges[edge].fromNode = fromNode;
    terrain->edges[edge].toNode = toNode;
    terrain->edges[edge].length = 
        glm::distance(terrain->nodes[fromNode].position, terrain->nodes[toNode].position);
    terrain->edges[edge].radiiRatio = 
        terrain->nodes[toNode].radius / terrain->nodes[fromNode].radius;
    terrain->edges[edge].direction = 
        glm::normalize(terrain->nodes[toNode].position - terrain->nodes[fromNode].position);
}

void updateNodeAdjcacencyPtrs(Terrain* terrain, int node, int start, int end, int prev)
{
    terrain->nodes[node].firstEdge = start;
    terrain->nodes[node].lastEdge = end;
    terrain->nodes[node].previousEdge = prev;
}

void updateModule(Terrain* terrain, int module, int startNode, int lastNode, int prevNode, int startEdge, int lastEdge)
{
    terrain->modules[module].startNode = startNode;
    terrain->modules[module].lastNode = lastNode;
    terrain->modules[module].previousNode = prevNode;
    terrain->modules[module].startEdge = startEdge;
    terrain->modules[module].lastEdge = lastEdge;
}

/** Create test tree (see images folder) */
void Tree::AddTree1(Terrain* terrain, glm::vec3 rootPos, float rootRadius)
{
    // get the starting indices
    int startModuleInx = terrain->modules.size();
    int startModuleEdgesInx = terrain->moduleEdges.size();
    int startNodeInx = terrain->nodes.size();
    int startEdgeInx = terrain->edges.size();

    // append the modules
    const int NUM_OF_MODULES = 3;
    for (int i = 0; i < NUM_OF_MODULES; i++)
    {
        Module module;
        terrain->modules.push_back(module);
    }

    // append the nodes
    const int NUM_OF_NODES = 22;
    for (int i = 0; i < NUM_OF_NODES; i++)
    {
        Node node;
        terrain->nodes.push_back(node);
    }

    // append the edges
    const int NUM_OF_EDGES = NUM_OF_NODES - 1;
    for (int i = 0; i < NUM_OF_EDGES; i++)
    {
        Edge edge;
        terrain->edges.push_back(edge);
    }

    // Connect the modules
    ModuleEdge moduleEdge;

    // - module 0's adjacency list
    terrain->modules[startModuleInx].startModule = startModuleEdgesInx;
    moduleEdge.moduleInx = startModuleInx + 1; // module 1
    terrain->moduleEdges.push_back(moduleEdge);
    moduleEdge.moduleInx = startModuleInx + 2; // module 2
    terrain->moduleEdges.push_back(moduleEdge);
    terrain->modules[startModuleInx].endModule = startModuleInx + 2;

    // - module 1's adjacency list
    terrain->modules[startModuleInx + 1].startModule = startModuleEdgesInx + 2;
    moduleEdge.moduleInx = startModuleInx;
    terrain->moduleEdges.push_back(moduleEdge); // module 0
    terrain->modules[startModuleInx + 1].endModule = startModuleEdgesInx + 2;

    // - module 2's adjacency list
    terrain->modules[startModuleInx + 2].startModule = startModuleEdgesInx + 3;
    moduleEdge.moduleInx = startModuleInx;
    terrain->moduleEdges.push_back(moduleEdge); // module 0
    terrain->modules[startModuleInx + 2].endModule = startModuleEdgesInx + 3;

    // Fill in node information
    const float BASE_RANDOM = 0.1;
    const float BRANCH_RANDOM = 0.4;
    // - module 0
    // -- node 0
    terrain->nodes[startNodeInx + 0].position = rootPos;
    terrain->nodes[startNodeInx + 0].radius = rootRadius;
    // -- node 1
    float length = 2.5;
    glm::vec3 dir = glm::normalize(glm::vec3(rnum(BASE_RANDOM), 1, rnum(BASE_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 1, startNodeInx + 0);
    // -- node 2
    length = 2;
    dir  = glm::normalize(glm::vec3(rnum(BASE_RANDOM), 1, rnum(BASE_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 2, startNodeInx + 1);
    // -- node 3
    length = 1.5;
    dir  = glm::normalize(glm::vec3(rnum(BASE_RANDOM), 1, rnum(BASE_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 3, startNodeInx + 2);
    // -- node 4
    length = 1;
    dir  = glm::normalize(glm::vec3(-1, rnum(BRANCH_RANDOM), rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 4, startNodeInx + 3);
    // -- node 5
    length = 1;
    dir = glm::normalize(glm::vec3(-1, rnum(BRANCH_RANDOM), rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 5, startNodeInx + 4);
    // -- node 6
    length = 0.8;
    dir = glm::normalize(glm::vec3(rnum(BRANCH_RANDOM), 1, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 6, startNodeInx + 4);
    // -- node 7
    length = 0.8;
    dir = glm::normalize(glm::vec3(rnum(BRANCH_RANDOM), 1, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 7, startNodeInx + 3);
    // -- node 8 
    length = 1;
    dir = glm::normalize(glm::vec3(1, rnum(BRANCH_RANDOM), rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 8, startNodeInx + 3);
    // -- node 9
    length = 0.8;
    dir = glm::normalize(glm::vec3(rnum(BRANCH_RANDOM), 1, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 9, startNodeInx + 8);
    // -- node 10
    length = 0.8;
    dir = glm::normalize(glm::vec3(rnum(BRANCH_RANDOM), -1, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 10, startNodeInx + 8);
    // -- node 11
    length = 1;
    dir = glm::normalize(glm::vec3(1, rnum(BRANCH_RANDOM), rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 11, startNodeInx + 8);
    
    // - module 1
    // -- node 12
    length = 0.8;
    dir = glm::normalize(glm::vec3(-0.5, 0.5, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 12, startNodeInx + 5);
    // -- node 13
    length = 0.5;
    dir = glm::normalize(glm::vec3(-1, rnum(BRANCH_RANDOM), rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 13, startNodeInx + 12);
    // -- node 14
    length = 0.7;
    dir = glm::normalize(glm::vec3(rnum(BRANCH_RANDOM), 1, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 14, startNodeInx + 12);
    // --node 15
    length = 0.5;
    dir = glm::normalize(glm::vec3(-1, rnum(BRANCH_RANDOM), rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 15, startNodeInx + 14);
    // -- node 16
    length = 0.4;
    dir = glm::normalize(glm::vec3(rnum(BRANCH_RANDOM), 1, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 16, startNodeInx + 14);
    
    // - module 2
    // -- node 17
    length = 0.7;
    dir = glm::normalize(glm::vec3(rnum(BRANCH_RANDOM), 1, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 17, startNodeInx + 11);
    // -- node 18
    length = 0.8;
    dir = glm::normalize(glm::vec3(1, rnum(BRANCH_RANDOM), rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 18, startNodeInx + 11);
    // -- node 19
    length = 0.7;
    dir = glm::normalize(glm::vec3(0.5, 0.5, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 19, startNodeInx + 18);
    // -- node 20
    length = 0.7;
    dir = glm::normalize(glm::vec3(1, rnum(BRANCH_RANDOM), rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 20, startNodeInx + 18);
    // -- node 21
    length = 0.7;
    dir = glm::normalize(glm::vec3(0.5, 0.5, rnum(BRANCH_RANDOM)));
    addNode(terrain, length, dir, startNodeInx + 21, startNodeInx + 18);
    
    // Connect edges
    addEdge(terrain, startEdgeInx + 0, startNodeInx + 0, startNodeInx + 1); // edge 0
    addEdge(terrain, startEdgeInx + 1, startNodeInx + 1, startNodeInx + 2); // edge 1
    addEdge(terrain, startEdgeInx + 2, startNodeInx + 2, startNodeInx + 3); // edge 2
    addEdge(terrain, startEdgeInx + 3, startNodeInx + 3, startNodeInx + 4); // edge 3
    addEdge(terrain, startEdgeInx + 4, startNodeInx + 3, startNodeInx + 7); // edge 4
    addEdge(terrain, startEdgeInx + 5, startNodeInx + 3, startNodeInx + 8); // edge 5
    addEdge(terrain, startEdgeInx + 6, startNodeInx + 4, startNodeInx + 5); // edge 6
    addEdge(terrain, startEdgeInx + 7, startNodeInx + 4, startNodeInx + 6); // edge 7
    addEdge(terrain, startEdgeInx + 8, startNodeInx + 8, startNodeInx + 9); // edge 8
    addEdge(terrain, startEdgeInx + 9, startNodeInx + 8, startNodeInx + 10); // edge 9
    addEdge(terrain, startEdgeInx + 10, startNodeInx + 8, startNodeInx + 11); // edge 10
    addEdge(terrain, startEdgeInx + 11, startNodeInx + 5, startNodeInx + 12); // edge 11
    addEdge(terrain, startEdgeInx + 12, startNodeInx + 12, startNodeInx + 13); // edge 12
    addEdge(terrain, startEdgeInx + 13, startNodeInx + 12, startNodeInx + 14); // edge 13
    addEdge(terrain, startEdgeInx + 14, startNodeInx + 14, startNodeInx + 15); // edge 14
    addEdge(terrain, startEdgeInx + 15, startNodeInx + 14, startNodeInx + 16); // edge 15
    addEdge(terrain, startEdgeInx + 16, startNodeInx + 11, startNodeInx + 17); // edge 16
    addEdge(terrain, startEdgeInx + 17, startNodeInx + 11, startNodeInx + 18); // edge 17
    addEdge(terrain, startEdgeInx + 18, startNodeInx + 18, startNodeInx + 19); // edge 18
    addEdge(terrain, startEdgeInx + 19, startNodeInx + 18, startNodeInx + 20); // edge 19
    addEdge(terrain, startEdgeInx + 20, startNodeInx + 18, startNodeInx + 21); // edge 20

    // update nodes' adjacency list pointers
    updateNodeAdjcacencyPtrs(terrain, 0, startEdgeInx + 0, startEdgeInx + 0, -1); // node 0
    updateNodeAdjcacencyPtrs(terrain, 1, startEdgeInx + 1, startEdgeInx + 1, 0); // node 1
    updateNodeAdjcacencyPtrs(terrain, 2, startEdgeInx + 2, startEdgeInx + 2, 1); // node 2
    updateNodeAdjcacencyPtrs(terrain, 3, startEdgeInx + 3, startEdgeInx + 5, 2); // node 3
    updateNodeAdjcacencyPtrs(terrain, 4, startEdgeInx + 6, startEdgeInx + 7, 3); // node 4
    updateNodeAdjcacencyPtrs(terrain, 5, startEdgeInx + 11, startEdgeInx + 6, 6); // node 5
    updateNodeAdjcacencyPtrs(terrain, 6, -1, -1, 7); // node 6
    updateNodeAdjcacencyPtrs(terrain, 7, -1, -1, 4); // node 7
    updateNodeAdjcacencyPtrs(terrain, 8, startEdgeInx + 8, startEdgeInx + 10, 5); // node 8
    updateNodeAdjcacencyPtrs(terrain, 9, -1, -1, 8); // node 9
    updateNodeAdjcacencyPtrs(terrain, 10, -1, -1, 9); // node 10
    updateNodeAdjcacencyPtrs(terrain, 11, startEdgeInx + 16, startEdgeInx + 17, 10); // node 11
    updateNodeAdjcacencyPtrs(terrain, 12, startEdgeInx + 12, startEdgeInx + 13, 11); // node 12
    updateNodeAdjcacencyPtrs(terrain, 13, -1, -1, 12); // node 13
    updateNodeAdjcacencyPtrs(terrain, 14, startEdgeInx + 14, startEdgeInx + 15, 13); // node 14
    updateNodeAdjcacencyPtrs(terrain, 15, -1, -1, 14); // node 15
    updateNodeAdjcacencyPtrs(terrain, 16, -1, -1, 15); // node 16
    updateNodeAdjcacencyPtrs(terrain, 17, -1, -1, 16); // node 17
    updateNodeAdjcacencyPtrs(terrain, 18, startEdgeInx + 18, startEdgeInx + 20, 17); // node 18
    updateNodeAdjcacencyPtrs(terrain, 19, -1, -1, 18); // node 19
    updateNodeAdjcacencyPtrs(terrain, 20, -1, -1, 18); // node 20
    updateNodeAdjcacencyPtrs(terrain, 21, -1, -1, 18); // node 21

    // update modules pointers 
    updateModule(
        terrain, 
        startModuleInx + 0, 
        startNodeInx + 0, 
        startNodeInx + 10, 
        -1, 
        0, 
        startEdgeInx + 10);
    updateModule(
        terrain, 
        startModuleInx + 1, 
        startNodeInx + 5, 
        startNodeInx + 16, 
        startNodeInx + 4, 
        startEdgeInx + 11, 
        startEdgeInx + 15);
    updateModule(
        terrain, 
        startModuleInx + 2, 
        startNodeInx + 11, 
        startNodeInx + 21, 
        startNodeInx + 8, 
        startEdgeInx + 16, 
        startEdgeInx + 20);
}
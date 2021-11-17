#include "Tree.h"

Terrain::Terrain(int x, int y) :
    x(x), y(y)
{}

int Terrain::AddTree(glm::vec3 pos, float radius)
{
    if (radius < 0)
        return -1;

    InternalNode rootNode;
    rootNode.radius = radius;
    rootNode.pos = position;

    branches.push_back(rootNode);

    ConnectionNode rootModule;
    rootModule.previousNode = -1;
    rootModule.rootNode = branches.size() - 1;

    rootModule = modules.push_back(module);
    modules.push_back(module);
    return rootModule;
}

int Terrain::AppendBranch(int node, float radius, float length, glm::vec3 dir, bool newModule)
{
    // Verify input is valid 
    if (node < 0 || node >= branches.size())
        return false;
    if (radius < 0 || length < 0)
        return false;
    dir = glm::normalize(dir); 

    InternalNode &branch = branches.at(node);

    // Create new node
    InternalNode newBranch;
    newBranch.radius = radius;
    newBranch.position = branch.position + dir * length;

    // create the new edge 
    Edge newEdge;
    newEdge.fromNode = node;
    newEdge.toNode = branches.size() - 1;
    newEdge.length = length;
    newEdge.radiiRatio = radius / branch.radius;
    newEdge.direction = dir;

    // Add to edges array and update the from node
    branch.lastEdge++;
    connections.insert(connections.begin() + lastEdge, newEdge);

    // update the to node
    newBranch.previousEdge = branch.lastEdge;

    if (newModule)
    {
        ConnectionNode module;
        module.previousNode = node;
        module.rootNode = branches.size() - 1;

        modules.push_back(module);
        return modules.size() - 1;
    }
    return branches.size() - 1;
}
#pragma once

#include <glm/glm.hpp>
#include "tree/Tree.h"

class Terrain
{
public:

/** Create a 2D plane with provided width and height*/
Terrain(int width, int height);

/** Currently only adds the same, basic tree at a target location */
void AddTree(int x, int y);

private:
int width, height;

// this will be passed to the device 
std::vector<InternalNode> nodes;         // representing branches within module
std::vector<Edge> edges;                 // link between branches within module
std::vector<ConnectionNode> connections; // link between modules 
};
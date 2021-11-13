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
std::vector<std::unique_ptr<Tree>> trees;
int width, height;
};
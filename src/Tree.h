#pragma once
#include "Terrain.h"

class Tree
{
public:
    Tree();
    ~Tree();

    void AddTree1(Terrain* terrain, glm::vec3 rootPos, float rootRadius);
};
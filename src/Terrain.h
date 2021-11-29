#pragma once
#include "TerrainStructs.h"
#include "Triangle.h"
#include <vector>

class Terrain
{
public:
    Terrain();
    ~Terrain();

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<Module> modules;

    Geom grass;
    Geom dirt;
};
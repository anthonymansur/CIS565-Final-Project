#pragma once
#include "TerrainStructs.h"
#include "Triangle.h"
#include <vector>
#include <sstream>
#include <fstream>

class Terrain
{
public:
    Terrain();
    ~Terrain();

    bool loadScene(std::string filename);

    float precipitation;
    float temperature;
    float ecosystemSize;

    float gridSideLength;

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<Module> modules;
    std::vector<ModuleEdge> moduleEdges;
};
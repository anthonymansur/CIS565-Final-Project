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
    bool loadTestScene();

    float precipitation;
    float temperature;
    float ecosystemSize;

    float gridSideLength;

    int numberOfTrees;

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<Module> modules;
    std::vector<ModuleEdge> moduleEdges;
};
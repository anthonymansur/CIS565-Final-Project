#include "Terrain.h"
#include <iostream>

Terrain::Terrain(int width, int height) : width(width), height(height)
{}

void Terrain::AddTree(int x, int y)
{
    // See Figure 2 in the research paper to see what I'm trying to do here

    // Create the tree to place in the terrain
    std::unique_ptr<Tree> tree = std::make_unique<Tree>(trees.size());
    tree->setTransformation(glm::vec3(0), 0);

    /** blue module */
    // Create one of the modules the tree will have 
    Module* blueModule = tree.get()->AddModule(glm::vec3(0,0,0), glm::vec3(0,1,0));

    // Create all the cones that module is made up of, with it's starting radius, ending radius, and length
    TruncatedCone blueCone1(0.01, 0.01, 0.1);
    TruncatedCone blueCone2(0.01, 0.01, 0.5);
    TruncatedCone blueCone3(0.01, 0.01, 0.4);

    // Orient and position the cones into its module
    blueModule->AddNode(blueCone1, glm::vec3(0, 0, 0), glm::normalize(glm::vec3(0.1, 0.9, 0)));
    blueModule->AddNode(blueCone2, glm::vec3(/*TODO*/), glm::normalize(glm::vec3(0.2, 0.8, 0)));
    blueModule->AddNode(blueCone3, glm::vec3(/*TODO*/), glm::normalize(glm::vec3(-0.2, 0.8, 0)));

    // connect the nodes to each other
    blueModule->AddEdge(0, 1);
    blueModule->AddEdge(0, 2);

    /** green module */
    Module* greenModule = tree.get()->AddModule(glm::vec3(/*TODO*/), glm::vec3(/*TODO*/));
    TruncatedCone greenCone1(0, 0, 0 /*TODO*/);
    greenModule->AddNode(greenCone1, glm::vec3(0, 0, 0), glm::vec3(/*TODO*/));

    /** gray module */
    Module* grayModule = tree.get()->AddModule(glm::vec3(/*TODO*/), glm::vec3(/*TODO*/));

    /** orange module */
    Module* orangeModule = tree.get()->AddModule(glm::vec3(/*TODO*/), glm::vec3(/*TODO*/));


    // connect the modules together
    bool moduleConnected = tree->connectModules(0, 1, 1, 0); 

    if (moduleConnected)
        std::cout << "We've been able to generate a tree!" << std::endl;
    else
        std::cout << "Oh no! There was an error generated trees..." << std::endl;
}
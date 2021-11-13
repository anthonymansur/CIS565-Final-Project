#pragma once

#include "Module.h"

struct ConnectionNode
{
    Module* fromModule, * toModule;
    InternalNode* fromNode, * toNode;

    ConnectionNode(Module* fromModule, Module* toModule, InternalNode* fromNode, InternalNode* toNode) :
        fromModule(fromModule), toModule(toModule), fromNode(fromNode), toNode(toNode)
    {}
};

class Tree
{
public:
    Tree(int id);

    /** Creates a new module in the tree and returns its key */
    Module* AddModule(glm::vec3 pos, glm::vec3 dir);

    /** Connect to modules in the tree */
    bool connectModules(int fromModule, int toModule, int fromNode, int toNode);

    inline void setTransformation(glm::vec3 pos, float ori) { position = pos; orientation = ori; };
private:
    std::vector<std::unique_ptr<Module>> modules;
    std::vector<ConnectionNode> connections;

    Module* rootModule;

    glm::vec3 position; // position of tree in world space
    float orientation;  // orientation of the tree (about the z-axis)

    int id; // unique id of the tree 
};
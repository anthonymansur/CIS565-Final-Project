#pragma once
#include <memory>
#include "TerrainStructs.h"

class Terrain
{
public:
    /** Initialize the terrain of specified size*/
    Terrain(int x, int y);

    /**
     * @brief Intializes the tree and returns its rootModule
     * @param pos the location of the base of the tree
     * @param radius the radius of the tree trunk
     * 
     * @return the index of the tree's root module, or -1 if error occured
     **/
    int AddTree(glm::vec3 pos, float radius);

    /**
     * @brief Append a new module to the specified terminal node
     * @param node the index of the terminal node of the branch 
     * 
     * @return the index of the new module, or -1 if error
     */
    int AppendModule(int moduleInx, int branchInx, float radius);

    /**
     * @brief Append a new branch to the specified module
     * @param node the index of the node to append the branch to
     * @param radius the starting radius of the branch
     * @param length the length of the branch
     * @param dir the direction the branch points too
     * @param newModule whether this creates a new module or not
     * 
     * @return if a new Module, returns the index of the new module, else
     * returns the index of the new branch created, or -1 if error
     */
    int AppendBranch(int node, float radius, float length, glm::vec3 dir, bool newModule);

private:
    int x, y;

    std::vector<InternalNode> branches;
    std::vector<Edge> connections;
    std::vector<ConnectionNode> modules;
};
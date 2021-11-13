#pragma once

#include <vector>
#include <array>
#include <glm/glm.hpp>

#include "TruncatedCone.h"

#include <memory>

struct InternalNode
{
  glm::vec3 position;                 // location in module space
  glm::vec3 direction;                // normalized direction the cone points to
  TruncatedCone branch;               // Reference to the TruncatedCone
  InternalNode* parentNode;           // parent node
  std::array<InternalNode*, 3> childrenNodes; // children nodes (node can't have more than three children)

  // TODO: if algorithm requires more than 3 and the maximum number is known at compile
  // time, replace it here and update implementation. Otherwise, a dynamically allocated
  // array needs to be stored and internal node must be placed in the heap.

  InternalNode(const TruncatedCone& cone, glm::vec3 pos, glm::vec3 dir) : 
	  position(pos), direction(dir), branch(TruncatedCone(cone)), parentNode(nullptr)
  {
	  this->childrenNodes[0] = nullptr;
	  this->childrenNodes[1] = nullptr;
	  this->childrenNodes[2] = nullptr;
  }

  bool canAddChild()
  {
	  return childrenNodes[2] == nullptr;
  }

  void addChild(InternalNode* child)
  {
	  if (childrenNodes[0] == nullptr)
		  childrenNodes[0] = child;
	  else if (childrenNodes[1] == nullptr)
		  childrenNodes[1] = child;
	  else
		  childrenNodes[2] = child;
  }
};

class Module
{
public:
	Module(int id);

	/** Creates a new node in the module and returns its key */
	int AddNode(const TruncatedCone& cone, glm::vec3 pos, glm::vec3 dir);

	/** Add a connection between two internal nodes */
	bool AddEdge(int fromNode, int toNode);

	/** Set the position and orientation of module */
	inline void SetTransformation(glm::vec3 pos, glm::vec3 dir) { position = pos; direction = dir; };

	InternalNode* GetNode(int inx);

	inline const InternalNode* getRootNode() const { return rootNode; }
	inline const Module* getParent() const { return parent; }
	inline const Module* getChild() const { return child; }

	inline void setRootNode(InternalNode* root) { rootNode = root; }
	inline void setParent(Module* parent) { this->parent = parent; }
	inline void setChild(Module* child) { this->child = child; }


private:
	/** All the truncated cones that make up the module */
	std::vector<InternalNode> branches;
	/** Edges between truncated cones */
	std::vector<std::array<InternalNode*, 2>> connections;
	/** Pointer to the internal node deemed the root node */
	InternalNode* rootNode;

	/** Pointer to the two connection nodes, stored by the Tree */
	Module *parent, *child;

	/** Id of the Module, with respect to the tree it's located in */
	int id;

	glm::vec3 position;  // position of module in tree-space
	glm::vec3 direction; // direction of module in tree space
};
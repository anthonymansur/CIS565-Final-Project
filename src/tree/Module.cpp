#include "Module.h"

Module::Module(int i) : id(i), rootNode(nullptr), parent(nullptr), child(nullptr)
{}

int Module::AddNode(const TruncatedCone& cone, glm::vec3 pos, glm::vec3 dir)
{
	InternalNode internalNode(cone, pos, dir);

	branches.push_back(internalNode);

	if (rootNode == nullptr)
		rootNode = &branches.at(0);

	return (int)(branches.size() - 1);
}

// TODO: Needs to be updated according to 4.2.1 description, which states that an edge contains
// the length of branch and the radii ratio.
bool Module::AddEdge(int fromNode, int toNode)
{

	if (fromNode < 0 || fromNode >= branches.size() || toNode < 0 || toNode >= branches.size())
		return false;

	InternalNode& internalFromNode = branches.at(fromNode);
	InternalNode& internalToNode = branches.at(toNode);

	if (internalToNode.parentNode != nullptr|| !internalFromNode.canAddChild())
		return false; 

	internalFromNode.addChild(&internalToNode);
	internalToNode.parentNode = &internalFromNode;
	connections.push_back(std::array<InternalNode*, 2> { &internalFromNode, &internalToNode });

	return true;
}

InternalNode* Module::GetNode(int inx)
{
	if (inx < 0 || inx >= branches.size())
		return nullptr;
	return &branches.at(inx);
}
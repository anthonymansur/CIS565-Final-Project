#include "Tree.h"

Tree::Tree(int id) : id(id), rootModule(nullptr)
{}

Module* Tree::AddModule(glm::vec3 pos, glm::vec3 dir)
{
	std::unique_ptr<Module> module = std::make_unique<Module>(modules.size());
	module->SetTransformation(pos, dir);

	modules.push_back(std::move(module));
	return modules.back().get();
}

bool Tree::connectModules(int fromModule, int toModule, int fromNode, int toNode)
{
	if (fromModule < 0 || toModule < 0 || fromModule >= modules.size() || toModule >= modules.size())
		return false;
	
	Module* fromModulePtr = modules.at(fromModule).get();
	Module* toModulePtr = modules.at(toModule).get();

	if (fromModulePtr->getChild() != nullptr || toModulePtr->getParent() != nullptr)
		return false;

	InternalNode* fromNodePtr = fromModulePtr->GetNode(fromNode);
	InternalNode* toNodePtr = toModulePtr->GetNode(toNode);

	if (fromNodePtr == nullptr || toNodePtr == nullptr)
		return false; 

	if (!fromNodePtr->canAddChild() || toNodePtr->parentNode != nullptr)
		return false;

	fromModulePtr->setChild(toModulePtr);
	toModulePtr->setParent(fromModulePtr);
	connections.push_back(ConnectionNode(fromModulePtr, toModulePtr, fromNodePtr, toNodePtr));

	return true;
}
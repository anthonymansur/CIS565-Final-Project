#include "Terrain.h"
#include "Tree.h"

Terrain::Terrain()
{
	Tree tree;
	tree.AddTree1(this, glm::vec3(0.f), .5f);
};

Terrain::~Terrain()
{};
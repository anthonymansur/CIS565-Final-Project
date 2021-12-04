#include "Terrain.h"
#include "Tree.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <array>
#include <chrono>
#include <iomanip>

#define _USE_MATH_DEFINES // Keep above math.h import
#include <math.h> 

#define FIXED_FLOAT(x) std::fixed <<std::setprecision(2)<<(x) 


Terrain::Terrain()
{};

Terrain::~Terrain()
{};

bool Terrain::loadTestScene()
{
	Tree tree;
	tree.AddTree1(this, glm::vec3(0.f), .5f);
	return true;
}

// taken from https://www.techiedelight.com/trim-string-cpp-remove-leading-trailing-spaces/
std::string trim(const std::string& s)
{
	auto start = s.begin();
	while (start != s.end() && std::isspace(*start)) {
		start++;
	}

	auto end = s.end();
	do {
		end--;
	} while (std::distance(start, end) > 0 && std::isspace(*end));

	return std::string(start, end + 1);
}

// taken from https://www.techiedelight.com/split-string-cpp-using-delimiter/
void tokenize(std::string const& str, const char delim,
	std::vector<std::string>& out)
{
	size_t start;
	size_t end = 0;

	while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
	{
		end = str.find(delim, start);
		out.push_back(str.substr(start, end - start));
	}
}

float coneMass(float density, float r0, float r1, float l)
{
  return density * (float)(M_PI / 3) * l * (r0*r0 + r0*r1 + r1*r1);
}

glm::vec4 coneCenterOfMass(glm::vec3 p0, glm::vec3 p1, float r0, float r1)
{
	float l = glm::distance(p0, p1);
	float height = (1 - (r1 / r0)) * 0.25 + (r1 / r0) * 0.5;
	height *= l;

	glm::vec3 dir = glm::normalize(p1 - p0);

	glm::vec3 pos = p0 + height * dir;

	return glm::vec4(pos.x, pos.y, pos.z, coneMass(500, r0, r1, l));
}

// TODO: the last node may not be added, so the last tree/module may not be added. 
bool Terrain::loadScene(std::string filename)
{
	std::ifstream scene;
	scene.open(filename);

	std::cout << "Loading Scene" << std::endl;

	std::string line;

	// skip comments in file
	std::vector<std::string> header;
	for (int i = 0; i < 39; i++)
	{
		std::getline(scene, line);
		tokenize(line, ':', header);
		if (i >= 5 && i <= 23)
		{
			switch (i)
			{
			case 5:
				precipitation = stof(trim(header.at(1)));
				break;
			case 6:
				temperature = stof(trim(header.at(1)));
				break;
			case 7:
				ecosystemSize = stof(trim(header.at(1)));
				break;
			}
		}
		header.clear();
	}
		
	std::vector<std::string> tokens;
	int lastSeenTree = 0;
	int lastSeenModule = 0;
	std::map<int, std::vector<int>> nodeMap;
	std::map<int, std::vector<int>> moduleMap;
	int moduleStartNode = 0;
	int treeStartModule = 0;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	bool stop = false;
	while (!stop)
	{
		stop = !std::getline(scene, line);
		// get the values of node
		tokenize(line, ' ', tokens);

		int treeId = -1;
		int moduleId = -1;
		int parent = -1;
		float posx = 0.f;
		float posy = 0.f;
		float posz = 0.f;
		float radius = 0.f;
		bool leaf = false;

		if (!stop)
		{
			treeId = stoi(tokens.at(0));
			moduleId = stoi(tokens.at(2));
			parent = stoi(tokens.at(3));
			posx = stof(tokens.at(5));
			posy = stof(tokens.at(6));
			posz = stof(tokens.at(7));
			radius = stof(tokens.at(8));
			leaf = (stoi(tokens.at(9)) == 1) ? true : false;
		}
		
		tokens.clear();

		bool updateModule = moduleId != lastSeenModule;
		bool updateTree = treeId != lastSeenTree;

		if (stop)
		{
			// make the next loop run to get the last tree in 
			updateModule = true;
			updateTree = true;
		}

		if (updateModule || updateTree)
		{
			/** new module */
			// update the adjacency list of each node in the module
			int startEdge = edges.size();
			for (std::map<int, std::vector<int>>::iterator it = nodeMap.begin(); it != nodeMap.end(); ++it) 
			{
				Node& prevNode = nodes[it->first];

				prevNode.firstEdge = edges.size();
				for (int nodeInx : it->second)
				{
					Edge edge;
					edge.fromNode = it->first;
					edge.toNode = nodeInx;
					edge.culled = false;

					Node& node = nodes[nodeInx];
					edge.length = glm::distance(prevNode.position, node.position);
					edge.radiiRatio = node.radius / prevNode.radius;

					node.previousEdge = edges.size();
					edges.push_back(edge);
				}
				prevNode.lastEdge = edges.size() - 1;
			}
			nodeMap.clear();

			// add module to terrain
			Module module;
			module.startNode = moduleStartNode;
			module.lastNode = nodes.size() - 1;
			module.startEdge = startEdge;
			module.lastEdge = edges.size() - 1;
			module.culled = false;
			module.previousNode = -1;
			// update module's previous node
			Node& rootNode = nodes[moduleStartNode];
			// // if not root module of tree
			for (int i = treeStartModule; i < modules.size(); i++)
			{
				// go through every module in the tree to see where this module
				Module& module = modules[i];
				bool found = false;
				for (int j = module.startNode; j <= module.lastNode; j++)
				{
					// go through every need in this module to see if we identify the connection node
					Node& node = nodes[j];
					if (glm::distance(rootNode.position, node.position) < FLT_EPSILON)
					{
						// connection node has been found
						found = true;
						module.previousNode = j;

						if (moduleMap.find(i) == moduleMap.end())
						{
							std::vector<int> list;
							list.push_back(modules.size());
							moduleMap.insert(std::pair<int, std::vector<int>>(i, list));
						}
						else
							moduleMap.at(i).push_back(modules.size());
						break;
					}
				}
				if (found) break;
			}

			modules.push_back(module);

			moduleStartNode = nodes.size();

			if (updateTree)
			{
				/** new tree */
				// update the adjacency list of each module in the tree
				int startEdge = edges.size();
				for (std::map<int, std::vector<int>>::iterator it = moduleMap.begin(); it != moduleMap.end(); ++it)
				{
					Module& prevModule = modules[it->first];

					prevModule.startModule = moduleEdges.size();
					for (int moduleInx : it->second)
					{
						ModuleEdge moduleEdge;
						moduleEdge.moduleInx = moduleInx;

						moduleEdges.push_back(moduleEdge);
					}
					prevModule.endModule = moduleEdges.size() - 1;

				}
				moduleMap.clear();
				treeStartModule = modules.size() - 1;

				//std::cout << "Working on tree: " << treeId << std::endl;
			}
			//std::cout << "Working on module: " << moduleId << std::endl;
		}

		if (stop) break;

		// create node
		Node node;
		node.position = glm::vec3(posx, posy, posz);
		node.radius = radius;
		node.firstEdge = -1;
		node.lastEdge = -1;
		node.previousEdge = -1;
		node.leaf = leaf;

		nodes.push_back(node);

		if (parent != -1)
		{
			// if not root node of module
			int prevNodeInx = parent;
			if (nodeMap.find(prevNodeInx) == nodeMap.end())
			{
				std::vector<int> list;
				list.push_back(nodes.size() - 1);
				nodeMap.insert(std::pair<int, std::vector<int>>(prevNodeInx, list));
			}
			else
				nodeMap.at(prevNodeInx).push_back(nodes.size() - 1);
		}

		// update last seen indices
		lastSeenTree = treeId;
		lastSeenModule = moduleId;
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Loading this scene took " << FIXED_FLOAT(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.f)<< " seconds." << std::endl;
	
	std::cout << "Scanning through forest to update some terrain parameters..." << std::endl;
	begin = std::chrono::steady_clock::now();

	// updating the bounding boxes of the modules
	for (int index = 0; index < modules.size(); index++)
	{
		Module& module = modules[index];
		glm::vec3 minPos{FLT_MAX}, maxPos{FLT_MIN};
		for (int i = module.startNode; i <= module.lastNode; i++)
		{
			// for every node in the module
			Node& node = nodes[i];
			glm::vec3 pos = node.position;
			for (int j = 0; j < 3; j++)
			{
				if (pos[j] < minPos[j])
					minPos[j] = pos[j];
				if (pos[j] > maxPos[j])
					maxPos[j] = pos[j];
			}
		}
		module.boundingMin = minPos;
		module.boundingMax = maxPos;
	} 

	// finding the average module side length
	float sum = 0.f;
	for (int index = 0; index < modules.size(); index++)
	{
		Module& module = modules[index];
		for (int i = 0; i < 2; i++)
		{
			sum += module.boundingMax[i] - module.boundingMin[i];
		}
	}
	gridSideLength = sum / (3 * modules.size());

	glm::vec3 minPos{ FLT_MAX }, maxPos{ FLT_MIN };
	for (int index = 0; index < nodes.size(); index++)
	{
		// for every node in the module
		Node& node = nodes[index];
		glm::vec3 pos = node.position;
		for (int j = 0; j < 3; j++)
		{
			if (pos[j] < minPos[j])
				minPos[j] = pos[j];
			if (pos[j] > maxPos[j])
				maxPos[j] = pos[j];
		}
	}

	std::cout << "The bounding box (x,y,z) of this scene is: (" << maxPos[0] - minPos[0] << ", "
		<< maxPos[1] - minPos[1] << ", " << maxPos[2] - minPos[2] << ") meters." << std::endl;

	for (int i = 0; i < modules.size(); i++)
	{
		Module& module = modules[i];
		std::vector<glm::vec4> coms;
		for (int j = module.startEdge; j <= module.lastEdge; j++)
		{
			Edge& edge = edges[j];
			Node& fromNode = nodes[edge.fromNode];
			Node& toNode = nodes[edge.toNode];

			coms.push_back(coneCenterOfMass(fromNode.position, toNode.position, fromNode.radius, toNode.radius));
		}
		glm::vec3 centerOfMass{0.f};
		float sumOfWeights = 0.f;
		for (glm::vec4 com : coms)
		{
			centerOfMass += com[3] * glm::vec3(com[0], com[1], com[2]);
			sumOfWeights += com[3];
		}

		module.centerOfMass = centerOfMass / sumOfWeights;
	}


	end = std::chrono::steady_clock::now();
	std::cout << "This process took " << FIXED_FLOAT(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.f)<< " seconds." << std::endl;
	
	scene.close();
	return true;
}
#include "Terrain.h"
#include "Tree.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <array>
#include <chrono>
#include <iomanip>

#define FIXED_FLOAT(x) std::fixed <<std::setprecision(2)<<(x) 


Terrain::Terrain()
{
	/*Tree tree;
	tree.AddTree1(this, glm::vec3(0.f), .5f);*/
};

Terrain::~Terrain()
{};

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

// TODO: the last node may not be added, so the last tree/module may not be added. 
bool Terrain::loadScene(std::string filename)
{
	std::ifstream scene;
	scene.open(filename);

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
			case 7:
				ecosystemSize = stof(trim(header.at(1)));
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

		if (!stop)
		{
			treeId = stoi(tokens.at(0));
			moduleId = stoi(tokens.at(2));
			parent = stoi(tokens.at(3));
			posx = stof(tokens.at(5));
			posy = stof(tokens.at(6));
			posz = stof(tokens.at(7));
			radius = stof(tokens.at(8));
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
			module.previousNode = -1;
			// update module's previous node
			Node& rootNode = nodes[moduleStartNode];
			// if not root module of tree
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
							list.push_back(modules.size() - 1);
							moduleMap.insert(std::pair<int, std::vector<int>>(i, list));
						}
						else
							moduleMap.at(i).push_back(modules.size() - 1);
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
	scene.close();
	return true;
}
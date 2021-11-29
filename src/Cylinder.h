#include <cstdlib>
#include <vector>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "utilityCore.hpp"
#include "glslUtility.hpp"

class Cylinder {
public:
	Cylinder(float base, float top, float ht, int sect, int stack);

	float baseRadius;
	float topRadius;
	float height;
	int sectorCount;
	int stackCount;
	std::vector<GLfloat> vertices;
	std::vector<float> normals;
	std::vector<float> texCoords;
	std::vector<float> circleVertices;
	std::vector<GLuint> indices;
	std::vector<int> lineIndices;
	std::vector<float> interleavedVertices;
	int stride;

	int baseIndex;
	int topIndex;

	void buildVertices();
	void clearArrays();

	void drawCylinder();
};
#pragma once
#include <cmath>
#include <vector>
#include <glm/glm.hpp>

enum GeomType {
	BRANCH,
	RECT,
	LEAF
};

struct Point {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec2 uv;
};

class Triangle {
public:
	Point p1;
	Point p2;
	Point p3;

	Triangle() {
		p1.pos = glm::vec3(0.f);
		p2.pos = glm::vec3(0.f);
		p3.pos = glm::vec3(0.f);
	}

	Triangle(Point point1, Point point2, Point point3) {
		p1 = point1;
		p2 = point2;
		p3 = point3;
	}
};

struct Geom {
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	glm::mat4 invTranspose;
	std::vector<Triangle> triangles;
	int num_verts;
	enum GeomType type;
};
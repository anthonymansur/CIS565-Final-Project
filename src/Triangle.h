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

//struct Triangle {
//	Point p1;
//	Point p2;
//	Point p3;
//};

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
//
//	Triangle(const Triangle& t) {
//		vtx[0] = t.vtx[0];
//		vtx[1] = t.vtx[1];
//		vtx[2] = t.vtx[2];
//	}
//
//	Triangle(unsigned int v0, unsigned int v1, unsigned int v2) {
//		vtx[0] = v0;
//		vtx[1] = v1;
//		vtx[2] = v2;
//	}
//
//	~Triangle()
//	{}
//
//	Triangle& operator= (const Triangle& t) {
//		vtx[0] = t.vtx[0];
//		vtx[1] = t.vtx[1];
//		vtx[2] = t.vtx[2];
//		return (*this);
//	}
//
//	unsigned int& operator[] (unsigned int i) {
//		return vtx[i];
//	}
//
//	unsigned int operator[] (unsigned int i) const {
//		return vtx[i];
//	}
//
//private:
//	unsigned int vtx[3];
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
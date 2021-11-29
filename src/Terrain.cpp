#include "Terrain.h"

Terrain::Terrain() {
	Point pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7;
	// hard coding ground
	pt0.pos = glm::vec3(-5.0f, 0.0f, 0.0f);
	pt1.pos = glm::vec3(5.0f, 0.0f, 0.0f);
	pt2.pos = glm::vec3(5.0f, 0.0f, -10.0f);
	pt3.pos = glm::vec3(-5.0f, 0.0f, -10.0f);

	/*pt4.pos = glm::vec3(-5.0f, -1.0f, 0.0f);
	pt5.pos = glm::vec3(5.0f, -1.0f, 0.0f);
	pt6.pos = glm::vec3(5.0f, -1.0f, -10.0f);
	pt7.pos = glm::vec3(-5.0f, -1.0f, -10.0f);*/

	//grass triangles
	Triangle t1 = Triangle(pt0, pt1, pt2);
	/*Triangle t2 = Triangle(pt2, pt0, pt3);
	Triangle t3 = Triangle(pt3, pt4, pt0);
	Triangle t4 = Triangle(pt0, pt4, pt5);
	Triangle t5 = Triangle(pt0, pt5, pt6);
	Triangle t6 = Triangle(pt6, pt0, pt1);
	Triangle t7 = Triangle(pt6, pt1, pt7);
	Triangle t8 = Triangle(pt7, pt1, pt2);
	Triangle t9 = Triangle(pt7, pt2, pt3);
	Triangle t10 = Triangle(pt3, pt7, pt4);
	Triangle t11 = Triangle(pt4, pt5, pt7);
	Triangle t12 = Triangle(pt7, pt6, pt5);*/

	grass.triangles.push_back(t1);
	/*grass.triangles.push_back(t2);
	grass.triangles.push_back(t3);
	grass.triangles.push_back(t4);
	grass.triangles.push_back(t5);
	grass.triangles.push_back(t6);
	grass.triangles.push_back(t7);
	grass.triangles.push_back(t8);
	grass.triangles.push_back(t9);
	grass.triangles.push_back(t10);
	grass.triangles.push_back(t11);
	grass.triangles.push_back(t12);*/

	grass.type = LEAF;
	grass.num_verts = 8;

	//grass.indices = { 0, 1, 2, 2, 0, 3, 3, 4, 0, 0, 4, 5, 0, 5, 6, 6, 0, 1, 6, 1, 7, 7, 1, 2, 7, 2, 3, 3, 7, 4, 4, 5, 7, 7, 6, 5 };
};

Terrain::~Terrain()
{};
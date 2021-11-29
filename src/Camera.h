#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <filesystem>
#include "utilityCore.hpp"
#include "glslUtility.hpp"

class Camera {
public:
	Camera(float aspectRatio);
	~Camera();

	bool leftMousePressed;
	bool rightMousePressed;
	double previousX, previousY;
	
	// controls
	float r, theta, phi;

	glm::mat4 view, proj, viewProj;
	glm::vec3 position;

	void UpdateOrbit(float deltaX, float deltaY, float deltaZ);
	void updateCamera(GLuint* program, int size = 1);
	void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	void mousePositionCallback(GLFWwindow* window, double xpos, double ypos, GLuint* program);
};
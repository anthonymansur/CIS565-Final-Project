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
	Camera();
	~Camera();

	const float fovy = (float)(PI / 4);
	const float zNear = 0.10f;
	const float zFar = 100.0f;
	int width = 1280;
	int height = 720;
	
	// controls
	bool leftMousePressed;
	bool rightMousePressed;
	double lastX;
	double lastY;
	float theta;
	float phi;
	float zoom;
	glm::mat4 projection;
	glm::vec3 position;

	void updateCamera(GLuint* program);
	void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	void mousePositionCallback(GLFWwindow* window, double xpos, double ypos, GLuint* program);

	glm::ivec2 resolution;
	glm::vec3 lookAt;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec2 pixelLength;
};
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
#include "physics/kernel.h"

// adapted from Boids homework

GLuint positionLocation = 0;
GLuint colorLocation = 1;
GLuint texcoordsLocation = 2;
const char* attributeLocations[] = { "Position", "Color", "Texcoords"};

GLuint PBO;
GLuint vboID[3];
GLuint displayImage;
//glslUtility::shaders_t shaders;

GLuint VAO;
GLuint treeVAO;
GLuint positions = 0;
GLuint IBO = 0;

GLuint program[2];
GLuint treeProgram[2];

const unsigned int PROG = 0;
const unsigned int T_PROG = 0;

const float fovy = (float)(PI / 4);
const float zNear = 0.10f;
const float zFar = 100.0f;
int width = 1280;
int height = 720;
int pointSize = 2; // might not need

// For camera controls
bool leftMousePressed = false;
bool rightMousePressed = false;
double lastX;
double lastY;
float theta = 1.22f;
float phi = -0.70f;
float zoom = 2.0f;
glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraPosition;

glm::mat4 projection;

// Main
const char* projectName;

int main(int argc, char* argv[]);

// Main Loop
void mainLoop();
void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void updateCamera();
void runCUDA();

// Init
bool init(int argc, char** argv);
void initVAO();
void initTextures();
void initShaders(GLuint* program);
void initPBO();
void initCuda();

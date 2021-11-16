#pragma once
#include <glm/glm.hpp>
#include <cstdlib>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define PI 3.141592653589793

GLuint program[2];

const unsigned int PROG = 0;

const float fovy = (float)(PI / 4);
const float zNear = 0.10f;
const float zFar = 10.0f;
int w = 1280;
int h = 720;
int pointSize = 2;

bool leftMousePressed = false;
bool rightMousePressed = false;
double lastX;
double lastY;
float theta = 1.22f;
float phi = -0.70f;
float zoom = 4.0f;
glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraPosition;

glm::mat4 projection;

void updateCamera();
void keyCall(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCall(GLFWwindow* window, int button, int action, int mods);
void mousePositionCall(GLFWwindow* window, double xpos, double ypos);

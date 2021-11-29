#include "Camera.h"

Camera::Camera() {
	theta = 1.22f;
	phi = -0.70f;
	zoom = 2.0f;
	leftMousePressed = false;
	rightMousePressed = false;
	lastX = 0.0;
	lastY = 0.0;
	lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
}

Camera::~Camera()
{}

void Camera::updateCamera(GLuint* program) {
	position.x = zoom * sin(phi) * sin(theta);
	position.z = zoom * cos(theta);
	position.y = zoom * cos(phi) * sin(theta);
	position += lookAt;

	projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	glm::mat4 view = glm::lookAt(position, lookAt, glm::vec3(0, 1, 0));
	projection = projection * view;

	GLint location;
	glUseProgram(program[0]);
	if ((location = glGetUniformLocation(program[0], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}

void Camera::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void Camera::mousePositionCallback(GLFWwindow* window, double xpos, double ypos, GLuint* program) {
	if (leftMousePressed) {
		// compute new camera parameters
		phi += (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
		updateCamera(program);
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
		updateCamera(program);
	}

	lastX = xpos;
	lastY = ypos;
}
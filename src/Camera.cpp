#include "Camera.h"

Camera::Camera(float aspectRatio) {
	r = 10.0f;
	theta = 0.0f;
	phi = 0.0f;
	view = glm::lookAt(glm::vec3(0.0f, 1.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 100.0f);
	//proj[1][1] *= -1; // y-coordinate is flipped
}

Camera::~Camera()
{}

void Camera::updateCamera(GLuint* program) {
	viewProj = proj * view;
	GLint location;
	glUseProgram(*program);
	if ((location = glGetUniformLocation(*program, "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &viewProj[0][0]);
	}
}

void Camera::UpdateOrbit(float deltaX, float deltaY, float deltaZ) {
	theta += deltaX;
	phi += deltaY;
	r = glm::clamp(r - deltaZ, 1.0f, 50.0f);

	float radTheta = glm::radians(theta);
	float radPhi = glm::radians(phi);

	glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), radTheta, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::rotate(glm::mat4(1.0f), radPhi, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 finalTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f)) * rotation * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, r));

	view = glm::inverse(finalTransform);
}

void Camera::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			leftMousePressed = true;
			glfwGetCursorPos(window, &previousX, &previousY);
		}
		else if (action == GLFW_RELEASE) {
			leftMousePressed = false;
		}
	}
	else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		if (action == GLFW_PRESS) {
			rightMousePressed = true;
			glfwGetCursorPos(window, &previousX, &previousY);
		}
		else if (action == GLFW_RELEASE) {
			rightMousePressed = false;
		}
	}
}

void Camera::mousePositionCallback(GLFWwindow* window, double xpos, double ypos, GLuint* program) {
	if (leftMousePressed) {
		double sensitivity = 0.3;
		float deltaX = static_cast<float>((previousX - xpos) * sensitivity);
		float deltaY = static_cast<float>((previousY - ypos) * sensitivity);

		UpdateOrbit(deltaX, deltaY, 0.0f);

		previousX = xpos;
		previousY = ypos;
		updateCamera(program);
	}
	else if (rightMousePressed) {
		double deltaZ = static_cast<float>((previousY - ypos) * 0.05);

		UpdateOrbit(0.0f, 0.0f, deltaZ);

		previousY = ypos;
		updateCamera(program);
	}
}
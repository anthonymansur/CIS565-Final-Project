#include "Camera.h"

void keyCall(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouseButtonCall(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCall(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
        // compute new camera parameters
        phi += (xpos - lastX) / w;
        theta -= (ypos - lastY) / h;
        theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
        updateCamera();
    }
    else if (rightMousePressed) {
        zoom += (ypos - lastY) / h;
        zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
        updateCamera();
    }

    lastX = xpos;
    lastY = ypos;
}

void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(w) / float(h), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG]);
    if ((location = glGetUniformLocation(program[PROG], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
}
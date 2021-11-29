// includes
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtx/transform.hpp>

#include "physics/kernel.h"
#include "Camera.h"
#include "Terrain.h"
#include "Triangle.h"

// definitions
#define FIXED_FLOAT(x) std::fixed <<std::setprecision(2)<<(x) 

#define DT 0.016 // in seconds

// variables
const char* projectName;
std::string deviceName;
GLFWwindow* window;
int width = 1280;
int height = 720;

Camera camera = Camera(width / (height * 1.f));
Terrain terrain = Terrain();
GLuint program[2];
const char* attributeLocations[] = { "Position" };
GLuint positionLocation = 1;
GLuint VAO;
GLuint PBO;
GLuint IBO;
GLuint VBO;
GLuint displayImage;
int num_triangles = 0;

const unsigned int PROG = 0;

// functions
bool init(int argc, char** argv);
void initShaders(GLuint* program);
void initVAO();
void mainLoop();
void runCUDA();

void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButton(GLFWwindow* window, int button, int action, int mods);
void mousePosition(GLFWwindow* window, double xpos, double ypos);

int main(int argc, char* argv[])
{
    projectName = "CIS-565 Final Project: Wildfire Simulation";

    if (init(argc, argv))
    {
        Terrain terrain;
        camera.UpdateOrbit(0, -25, -15);
        camera.updateCamera(&program[PROG]);
        // TODO: generate terrain
        Simulation::initSimulation(&terrain);
        mainLoop();
        return 0;
    }
    else
    {
        return 1;
    }
}

bool init(int argc, char** argv)
{
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count)
    {
        std::cout
        << "Error: GPU device number is greater than the number of devices!"
        << " Perhaps a CUDA-capable GPU is not installed?"
        << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name<< "]";
    deviceName = ss.str();

    // Window setup
    glfwSetErrorCallback(errorCallback);

    // TODO: customize cmake to include opengl 4.6. most likely have to switch from glew to glad
    // Talk to Anthony for this
    if (!glfwInit())
    {
        std::cout 
        << "Error: Could not initialize GLFW!"
        << " Perhaps OpenGL 3.3 isn't available?"
        << std::endl;
        return false;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    } 
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePosition);
    glfwSetMouseButtonCallback(window, mouseButton);

    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize shaders

    initShaders(program);
    initVAO();

    // camera setup
    camera.updateCamera(program);

    // **CUDA OpenGL Interoperability**

    // Default to device ID 0. If you have more than one PGU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    // register buffer objects here
    // TODO: impelment

    // GL enables go here 
    //glEnable(GL_DEPTH_TEST);

    return true;
}

void mainLoop()
{
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();

        // calculate fps 
        if (time - timebase > 1.0)
        {
            fps = frame / (time - timebase);
            timebase = time;
            frame = 0;
        }

        runCUDA();

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());

        // GL commands go here for visualization
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // black background
        glClear(GL_COLOR_BUFFER_BIT /* | GL_DEPTH_BUFFER_BIT */);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, displayImage);

        glUseProgram(program[PROG]);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

void runCUDA()
{
    /** Map buffer objects between CUDA and GL */
    // cudaGLMapBufferObject(XXX);

    /** Timing analysis? */
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);

    // // What you want to time goes here

    // cudaEventRecord(stop);

    // cudaEventSynchronize(stop);
    // float milliseconds = 0.f;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    
    // std::cout << "FPS: " << FIXED_FLOAT(1 / milliseconds) << std::endl;

    /** Unmap buffer objects */
    // cudaGLUnmapBufferObject(XXX);

    Simulation::stepSimulation(DT);
}

void initShaders(GLuint* program) {
    GLuint location;
    program[PROG] = glslUtility::createProgram(
        "shaders/shader.vert.",
        /*"shaders/graphics.geom.glsl",*/
        "shaders/shader.frag", attributeLocations, 1);
    glUseProgram(program[PROG]);

    //glBindVertexArray(VAO);

    if ((location = glGetUniformLocation(program[PROG], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &camera.viewProj[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &camera.position[0]);
    }
    
    //glm::mat4 model = glm::rotate(90.f, glm::vec3(1.f, 0.f, 0.f));
    glm::mat4 model = glm::mat4(1.f);
    if ((location = glGetUniformLocation(program[PROG], "u_model")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE , &model[0][0]);
    }
}

void initVAO() {
    
    GLfloat vertices[] =
    {
        -10.0f, 0.0f, -10.0f, // bottom left
        -10.0f, 0.0f, 10.0f, // top left
        10.0f, 0.f, 10.0f, // top right
        10.0f, 0.f, -10.0f // bottom right
    };

    GLushort indices[] =
    {
        0, 1, 2, 0, 2, 3
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &IBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0); //maybe change
    glEnableVertexAttribArray(0);
}

// GLFW Callbacks
void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }

  void mouseButton(GLFWwindow* window, int button, int action, int mods) {
      camera.mouseButtonCallback(window, button, action, mods);
  }

  void mousePosition(GLFWwindow* window, double xpos, double ypos) {
      camera.mousePositionCallback(window, xpos, ypos, program);
  }
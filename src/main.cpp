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

#include "physics/kernel.h"
#include "Camera.h"
#include "Terrain.h"
#include "Triangle.h"
#include "Cylinder.h"

// definitions
#define FIXED_FLOAT(x) std::fixed <<std::setprecision(2)<<(x) 

// variables
const char* projectName;
std::string deviceName;
GLFWwindow* window;
int width = 1280;
int height = 720;

Camera camera = Camera(width / (height * 1.f));
Terrain terrain = Terrain();
Cylinder branch = Cylinder(0.5, 0.3, 3.0, 36, 8);
std::vector<Geom> geoms;
GLuint program[2];
const char* attributeLocations[] = { "Normals", "Position", "TexCoords"};
GLuint positionLocation = 1;
GLuint VAO;
GLuint PBO;
GLuint IBO;
GLuint VBO;
GLuint displayImage;
int num_triangles = 0;
int num_verts = 0;

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
    //geoms.push_back(terrain.grass);

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
        /*glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);*/

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // black background
        glClear(GL_COLOR_BUFFER_BIT /* | GL_DEPTH_BUFFER_BIT */);

        /*glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, displayImage);*/

        glUseProgram(program[PROG]);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, (unsigned int)branch.indices.size(), GL_UNSIGNED_SHORT, 0);

        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

void runCUDA()
{
    // TODO: implement

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
}

void initShaders(GLuint* program) {
    GLuint location;
    program[PROG] = glslUtility::createProgram(
        "shaders/shader.vert.",
        /*"shaders/graphics.geom.glsl",*/
        "shaders/shader.frag", attributeLocations, 3);
    glUseProgram(program[PROG]);

    //glBindVertexArray(VAO);

    if ((location = glGetUniformLocation(program[PROG], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &camera.viewProj[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &camera.position[0]);
    }
}

void initVAO() {
    
    std::vector<GLfloat> vertices;
    std::vector<GLushort> indices;
    int offset = 0;

    for (Geom &g : geoms) {
        for (Triangle &t : g.triangles) {
            vertices.push_back(t.p1.pos.x);
            vertices.push_back(t.p1.pos.y);
            vertices.push_back(t.p1.pos.z);
            vertices.push_back(t.p2.pos.x);
            vertices.push_back(t.p2.pos.y);
            vertices.push_back(t.p2.pos.z);
            vertices.push_back(t.p3.pos.x);
            vertices.push_back(t.p3.pos.y);
            vertices.push_back(t.p3.pos.z);
            num_triangles++;
        }
        // fill index buffer
        if (g.type == RECT) {
            // ground
            for (int i = 1; i <= g.num_verts - 3; i++) {
                indices.push_back(0);
                indices.push_back(i);
                indices.push_back(i + 1);

                indices.push_back(g.num_verts - 1);
                indices.push_back(i);
                indices.push_back(i + 1);
            }
            indices.push_back(0);
            indices.push_back(g.num_verts - 2);
            indices.push_back(1);

            indices.push_back(g.num_verts - 1);
            indices.push_back(g.num_verts - 2);
            indices.push_back(1);
        }
        else if (g.type == BRANCH) {
            // top circle
            for (int i = 1; i < 6; i++) { //change 6 to sect number
                indices.push_back(0);
                indices.push_back(i);
                indices.push_back(i + 1);
            }
            //middle
            for (int i = 1; i < 6; i++) {
                indices.push_back(i);
                indices.push_back(i + 1);
                indices.push_back(i + 6); // some offset to correspond to number of bottom verts
            }
            //bottom circle
            

        }
        else if (g.type == LEAF) {
            // triangle leaves for now
            indices.push_back(0);
            indices.push_back(1);
            indices.push_back(2);
        }
    }

    num_verts = indices.size();

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &IBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * branch.interleavedVertices.size(), branch.interleavedVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * branch.indices.size(), branch.indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
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
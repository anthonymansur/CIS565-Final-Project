/**
* @file main.cpp
**/

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "main.hpp"
#include "cylinder.hpp"
#include "sceneStructs.h"
#include <iostream>
#include <filesystem>

// definitions
#define FIXED_FLOAT(x) std::fixed <<std::setprecision(2)<<(x) 

// variables
std::string deviceName;
GLFWwindow* window;
Cylinder tree(1.0f, 0.2f, 3.0f, 36, 8);

Geom geom;

const float DT = 0.2f;
const int N_FOR_VIS = 2;


int main(int argc, char* argv[])
{
    projectName = "CIS-565 Final Project: Wildfire Simulation";

    if (init(argc, argv))
    {
        // TODO: implement
        mainLoop();
        Simulation::endSimulation();
        return 0;
    }
    else
    {
        return 1;
    }
}

bool init(int argc, char** argv)
{
    //TESTING
    Point pt0, pt1, pt2;
    pt0.pos = glm::vec3(-5.0f, 0.0f, 0.0f);
    pt1.pos = glm::vec3(5.0f, 0.0f, 0.0f);
    pt2.pos = glm::vec3(5.0f, 0.0f, -10.0f);
    Triangle triangle(pt0, pt1, pt2);
    geom.triangles.push_back(triangle);


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
   // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    } 
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize drawing state
    initShaders(program);
    initVAO();
    initTextures();
    //initCuda();
    //initPBO();

    updateCamera();


   // glUseProgram(PROG[program]);
    //glActiveTexture(GL_TEXTURE0);

    // **CUDA OpenGL Interoperability**

    // register buffer objects here
    
    //cudaGLRegisterBufferObject(vboID[0]);
    //cudaGLRegisterBufferObject(vboID[1]);

   // Simulation::initSimulation(1);

    // GL enables go here 
   // glEnable(GL_DEPTH_TEST);

    return true;
}

void initVAO() {

    // pos (3), color (3), tex (2)
    /*GLfloat vertices[] = {
        -5.f, 0.0f, -10.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 
        -5.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 
        5.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 
        5.0f, 0.0f, -10.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f 
    };

    GLushort indices[] = { 0, 1, 2, 2, 0, 3 };*/

    GLfloat vertices[] = {
        -5.f, 0.0f, -10.0f,
        -5.0f, 0.0f, 0.0f,
        5.0f, 0.0f, 0.0f,
    };

    GLushort indices[] = { 0, 1, 2, 2, 0, 3 };

    // TESTING
   /* std::vector<GLfloat> vertices;
    std::vector<GLushort> indices;

    for (Triangle& t : geom.triangles) {
        vertices.push_back(t.p1.pos.x);
        vertices.push_back(t.p1.pos.y);
        vertices.push_back(t.p1.pos.z);
        vertices.push_back(t.p2.pos.x);
        vertices.push_back(t.p2.pos.y);
        vertices.push_back(t.p2.pos.z);
        vertices.push_back(t.p3.pos.x);
        vertices.push_back(t.p3.pos.y);
        vertices.push_back(t.p3.pos.z);
    }

    indices = { 0, 1, 2 };

    GLfloat* verticesData = vertices.data();
    GLushort* indexData = indices.data();*/

    //glGenBuffers(3, vboID);
    unsigned int VBO, idx;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &idx);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    ////color
    //glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    //glEnableVertexAttribArray(1);

    //// texcoord
    //glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    //glEnableVertexAttribArray(2);
    
   // glBindBuffer(GL_ARRAY_BUFFER, 0);
   // glBindVertexArray(0);

   /* glBindBuffer(GL_ARRAY_BUFFER, vboID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vboID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texCoords), texCoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);*/

}

void initShaders(GLuint* program) {
    GLuint location;
    program[PROG] = glslUtility::createProgram(
        "shaders/graphics.vert.glsl",
        /*"shaders/graphics.geom.glsl",*/
        "shaders/graphics.frag.glsl", attributeLocations, 1);
    glUseProgram(program[PROG]);

    //glBindVertexArray(VAO);

    // does this need to be changed?
    if ((location = glGetUniformLocation(program[PROG], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
}

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    
    //try to load image texture
    //GLint size;
    //char* img = glslUtility::loadFile("textures/tree.jpg", size);

    /*int w, h, c;
    unsigned char* img = stbi_load("./textures/tree.jpg", &w, &h, &c, 0);
    if (img) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else {
        std::cout << "Unable to open file " << std::endl;
    }
    int location = glGetUniformLocation(program[PROG], "ourTex");
    glUniform1i(location, 0);*/
}

void initPBO() {
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size = sizeof(GLubyte) * num_values;

    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, NULL, GL_DYNAMIC_COPY);

    cudaGLRegisterBufferObject(PBO);
}

void deletePBO(GLuint* pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda() {
    if (PBO) {
        deletePBO(&PBO);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void initCuda() {
    cudaGLSetGLDevice(0);
    atexit(cleanupCuda);
}

void runCUDA()
{
    // TODO: implement
    float4* dptr = NULL;
    float* dptrVertPositions = NULL;

    /** Map buffer objects between CUDA and GL */
    //cudaGLMapBufferObject((void**)&dptrVertPositions, positions);
    cudaGLMapBufferObject((void**)&dptrVertPositions, PBO);

    /** Timing analysis? */
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);

    Simulation::stepSimulation(DT);

    // cudaEventRecord(stop);

    // cudaEventSynchronize(stop);
    // float milliseconds = 0.f;
    // cudaEventElapsedTime(&milliseconds, start, stop);

    // std::cout << "FPS: " << FIXED_FLOAT(1 / milliseconds) << std::endl;

    /** Unmap buffer objects */
    cudaGLUnmapBufferObject(PBO);
}

void drawBranch() {
    glGenVertexArrays(1, &treeVAO);
    glBindVertexArray(treeVAO);
    tree.drawCylinder();
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
        //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
        //glBindTexture(GL_TEXTURE_2D, displayImage);
        //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT/* | GL_DEPTH_BUFFER_BIT*/);

        // texture
        //glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_2D, displayImage);

        glUseProgram(program[PROG]);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, 0);

        /*glBindVertexArray(treeVAO);
        drawBranch();*/

        glfwSwapBuffers(window);

        glUseProgram(0);
        glBindVertexArray(0);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
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

  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
      leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
      rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
      if (leftMousePressed) {
          // compute new camera parameters
          phi += (xpos - lastX) / width;
          theta -= (ypos - lastY) / height;
          theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
          updateCamera();
      }
      else if (rightMousePressed) {
          zoom += (ypos - lastY) / height;
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

      projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
      glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 1, 0));
      projection = projection * view;

      //glm::mat4 view = glm::mat4(1.0f);
      //glm::mat4 model = glm::mat4(1.0f);
      //model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
      //view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
      //projection = glm::perspective(fovy, (float)width / (float)height, zNear, zFar);
      //glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, -1));
      //projection = projection * view * model * glm::vec4(cameraPosition, 1.0);

      GLint location;

      glUseProgram(program[PROG]);
      if ((location = glGetUniformLocation(program[PROG], "u_projMatrix")) != -1) {
          glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
      }
  }
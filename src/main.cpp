/**
* @file main.cpp
**/

#include "main.hpp"

// definitions
#define FIXED_FLOAT(x) std::fixed <<std::setprecision(2)<<(x) 

// variables
std::string deviceName;
GLFWwindow* window;

const float DT = 0.2f;
const int N_FOR_VIS = 1;

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
    initCuda();
    initPBO();

    updateCamera();


   // glUseProgram(PROG[program]);
    //glActiveTexture(GL_TEXTURE0);

    // **CUDA OpenGL Interoperability**

    // register buffer objects here
    
    //cudaGLRegisterBufferObject(vboID[0]);
   // cudaGLRegisterBufferObject(vboID[1]);

    //Simulation::initSimulation(4);

    // camera setup
    
    //initShaders(program);

    // GL enables go here 
   // glEnable(GL_DEPTH_TEST);

    return true;
}

void initVAO() {

    //GLfloat vertices[] = {
    //    0.0f, 1.0f, 0.0f,
    //    25.0f, 1.0f, 0.0f,
    //    25.0f, 1.0f, 25.0f,
    //    0.0f, 1.0f, 25.0f,
    //};

    GLfloat vertices[] = {
       -1.0f, -1.0f, 0.0f,
       1.0f, -1.0f, 0.0f,
       1.0f, 1.0f, 0.0f,
       -1.0f, 1.0f, 0.0f,
    };

    GLfloat texCoords[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
    };

    /*GLfloat texCoords[] = {
        0.0f, 1.0f, 0.0f,
        25.0f, 1.0f, 0.0f,
        25.0f, 1.0f, 25.0f,
        0.0f, 1.0f, 25.0f,
    };*/

    GLushort indices[] = { 0, 1, 2, 2, 0, 3 };
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    //color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // texcoord
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    

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
        "shaders/graphics.frag.glsl", attributeLocations, 2);
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
    GLint size;
    //char* img = glslUtility::loadFile("textures\tree.jpg", size);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);
    int location = glGetUniformLocation(program[PROG], "ourTex");
    glUniform1i(location, 0);
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

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT/* | GL_DEPTH_BUFFER_BIT*/);

        // texture
       /*glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, displayImage);*/

        glUseProgram(program[PROG]);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        glfwSwapBuffers(window);

        /*glUseProgram(program[PROG]);
        glBindVertexArray(VAO);
        glPointSize((GLfloat)pointSize);
        glDrawElements(GL_TRIANGLES, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
        glPointSize(1.0f);

        glUseProgram(0);
        glBindVertexArray(0);*/


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
      glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, -1));
      projection = projection * view;

      GLint location;

      glUseProgram(program[PROG]);
      if ((location = glGetUniformLocation(program[PROG], "u_projMatrix")) != -1) {
          glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
      }
  }
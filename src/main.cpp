// includes
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
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
#include <array>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "physics/kernel.h"
#include "Camera.h"
#include "Terrain.h"

// definitions
#define FIXED_FLOAT(x) std::fixed <<std::setprecision(2)<<(x) 
#define DT 0.016 // in seconds
#define BIG_SCENE 

// variables
const char* projectName;
std::string deviceName;
GLFWwindow* window;
int width = 1280;
int height = 720;

Camera camera = Camera(width / (height * 1.f));
Terrain terrain;

GLuint program[3];
const char* attributeLocations_terrain[] = { "pos", "texCoords"};
GLuint VAO_terrain;
GLuint IBO_terrain;
GLuint VBO_terrain;

const char* attributeLocations_branches[] = { "v0", "v1", "attrib"};
GLuint VAO_branches;
GLuint VBO_branches;

const char* attributeLocations_fluid[] = { "pos", "col"};
GLuint VAO_smoke;
GLuint VBO_smoke;
GLuint IBO_smoke;

unsigned int texture;


const unsigned int PROG_terrain = 0;
const unsigned int PROG_branches = 2;
const unsigned int PROG_fluid = 1;

static float totalTime = 0.f;

// Grid Dimensions
#ifdef BIG_SCENE
int3 gridCount = { 88, 8, 24 };
float3 gridSize = { 220.f, 20.f, 60.f };
#else
int3 gridCount = { 24, 8, 24 };
float3 gridSize = { 60.f, 20.f, 60.f };
#endif
float sideLength = 2.5f; // "blockSize"

// GUI variables
static bool renderLeaves = false;
static bool renderModuleTemp = true;
static bool renderSmoke = false;

// functions
bool init(int argc, char** argv);
void initShaders(GLuint* program);
void initVAO(int NUM_OF_BRANCHES);
void initSmokeQuads();
void mainLoop(int NUM_OF_BRANCHES, std::ofstream& myfile);
void runCUDA();

void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButton(GLFWwindow* window, int button, int action, int mods);
void mousePosition(GLFWwindow* window, double xpos, double ypos);

int main(int argc, char* argv[])
{
    projectName = "CIS-565 Final Project: Wildfire Simulation";
    
    // Load the scene on which to run simulation on
    // TODO: Need to make updates so that updating scene is seamless
    #ifdef BIG_SCENE
    terrain.loadScene("../scenes/scene2.ply", gridCount.x, gridCount.y, gridCount.z, sideLength);
    #else
    terrain.loadScene("../scenes/scene1.ply", gridCount.x, gridCount.y, gridCount.z, sideLength);
    #endif

    // Forest Details
    std::cout << "Number of trees: " << terrain.numberOfTrees << std::endl;
    std::cout << "Number of modules: " << terrain.modules.size() << std::endl;
    std::cout << "Number of branches: " << terrain.edges.size() << std::endl;

    // Attempt to initialize application with CUDA and OpenGL
    if (init(argc, argv))
    {        
        // Initialize simulation
        camera.UpdateOrbit(0, -25, -15);
        camera.updateCamera(program, 3);
        Simulation::initSimulation(&terrain, gridCount);

        std::ofstream myfile;
        myfile.open("../fps.txt");

        // Run main loop
        mainLoop(terrain.edges.size(), myfile);

        myfile.close();

        // Exit
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

    // May want to upgrade OpenGL in the future
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

    // Initialize drawing state
    initVAO(terrain.edges.size());
    initSmokeQuads();

    // **CUDA OpenGL Interoperability**

    // Default to device ID 0. If you have more than one PGU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    // register buffer objects here
    cudaGLRegisterBufferObject(VBO_branches);
    cudaGLRegisterBufferObject(VBO_smoke);

    // Initialize shaders
    initShaders(program);

    // GL enables go here 
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);

    return true;
}

void mainLoop(int NUM_OF_BRANCHES, std::ofstream &myfile)
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

        // Run the simulation
        runCUDA();

        // Update window title 
        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());

        myfile << fps << std::endl;

        // GL commands go here for visualization
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        glBindTexture(GL_TEXTURE_2D, texture);

        /** Draw terrain */
        glUseProgram(program[PROG_terrain]);
        glBindVertexArray(VAO_terrain);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

        /** Draw branches */
        glUseProgram(program[PROG_branches]);
        glBindVertexArray(VAO_branches);
        glDrawArrays(GL_POINTS, 0, NUM_OF_BRANCHES);
        glPointSize(1.f);

        /** Draw smoke */
        if (renderSmoke)
        {
            glUseProgram(program[PROG_fluid]);
            glBindVertexArray(VAO_smoke);
            glDisable(GL_CULL_FACE);
            glClear(GL_DEPTH_BUFFER_BIT);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glDrawElements(GL_TRIANGLES, 3 * 6 * gridCount.x * gridCount.y * gridCount.z, GL_UNSIGNED_INT, 0);

            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
        }
        
        glBindVertexArray(0);
        glUseProgram(0);

        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

void runCUDA()
{
    /** Map buffer objects between CUDA and GL */
    float* dptrBranches = NULL;
    float* d_out = NULL;
    cudaGLMapBufferObject((void**)&dptrBranches, VBO_branches);
    cudaGLMapBufferObject((void**)&d_out, VBO_smoke);

    /** Timing analysis? */
     //cudaEvent_t start, stop;
     //cudaEventCreate(&start);
     //cudaEventCreate(&stop);

     //cudaEventRecord(start);
     // What you want to time goes here
    Simulation::stepSimulation(DT, gridCount, gridSize, sideLength, d_out);

    glUseProgram(program[PROG_fluid]);
    glBindVertexArray(VAO_smoke);
    totalTime += DT;
    GLuint location;
    if ((location = glGetUniformLocation(program[PROG_fluid], "u_time")) != -1) {
        glUniform1f(location, totalTime);
    }
    glBindVertexArray(0);
    glUseProgram(0);
     //cudaEventRecord(stop);

     //cudaEventSynchronize(stop);
     //float milliseconds = 0.f;
     //cudaEventElapsedTime(&milliseconds, start, stop);
    
     //std::cout << "FPS: " << FIXED_FLOAT(1 / milliseconds) << std::endl;

    Simulation::copyBranchesToVBO(dptrBranches);

    /** Unmap buffer objects */
    cudaGLUnmapBufferObject(VBO_branches);
    cudaGLUnmapBufferObject(VBO_smoke);
    cudaDeviceSynchronize();
}

void initShaders(GLuint* program) {
    GLuint location;
    program[PROG_terrain] = glslUtility::createProgram(
        "shaders/shader.vert",
        /*"shaders/graphics.geom.glsl",*/
        "shaders/shader.frag", attributeLocations_terrain, 2);
    program[PROG_branches] = glslUtility::createProgram(
        "shaders/branches.vert.glsl",
        "shaders/branches.geom.glsl",
        "shaders/branches.frag.glsl", attributeLocations_branches, 3);
    program[PROG_fluid] = glslUtility::createProgram(
        "shaders/fluid.vert",
        "shaders/fluid.frag", attributeLocations_fluid, 2); // len of the attributeLocations_fluid

    glUseProgram(program[PROG_terrain]);

    if ((location = glGetUniformLocation(program[PROG_terrain], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &camera.viewProj[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_terrain], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &camera.position[0]);
    }
    
    //glm::mat4 model = glm::rotate(90.f, glm::vec3(1.f, 0.f, 0.f));
    glm::mat4 model = glm::mat4(1.f);
    if ((location = glGetUniformLocation(program[PROG_terrain], "u_model")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE , &model[0][0]);
    }

    glUseProgram(program[PROG_branches]);

    if ((location = glGetUniformLocation(program[PROG_branches], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &camera.viewProj[0][0]);
    }

    if ((location = glGetUniformLocation(program[PROG_branches], "u_renderTemp")) != -1) {
        glUniform1i(location, renderModuleTemp);
    }
    if ((location = glGetUniformLocation(program[PROG_branches], "u_renderLeaves")) != -1) {
        glUniform1i(location, renderLeaves);
    }

    glUseProgram(program[PROG_fluid]);

    if ((location = glGetUniformLocation(program[PROG_fluid], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &camera.viewProj[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_fluid], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &camera.position[0]);
    }
    if ((location = glGetUniformLocation(program[PROG_fluid], "u_model")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &model[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_fluid], "u_renderSmoke")) != -1) {
        glUniform1i(location, renderSmoke);
    }
    if ((location = glGetUniformLocation(program[PROG_fluid], "u_time")) != -1) {
        glUniform1f(location, totalTime);
    }
}

int flatten(const int i_x, const int i_y, const int i_z) {
    return i_x + i_y * gridCount.x + i_z * gridCount.y * gridCount.z;
}

void recomputeIndices(int axis) {
    GLuint nflat = gridCount.x * gridCount.y * gridCount.z;
    const unsigned int numFloatsPerCell = (3 + 4) * 4;
    GLuint* smokeIndexes = new GLuint[6 * nflat]; // 2 triangles

    for (GLuint i = 0; i < 6 * nflat; i += 6) {
        GLuint offset = (i / 6) * 4 + (nflat * axis * numFloatsPerCell);
        smokeIndexes[i] = offset;
        smokeIndexes[i + 1] = offset + 1;
        smokeIndexes[i + 2] = offset + 2;
        smokeIndexes[i + 3] = offset;
        smokeIndexes[i + 4] = offset + 2;
        smokeIndexes[i + 5] = offset + 3;
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO_smoke);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * nflat * sizeof(GLuint), smokeIndexes, GL_DYNAMIC_DRAW);
}

void initSmokeQuads() {
    const unsigned int numFloatsPerCell = (3 + 4) * 4; // 3 floats from pos 4 floats from col, and we have 4 verts per quad
    GLuint nflat = gridCount.x * gridCount.y * gridCount.z;
    GLfloat* smokeQuadsPositions = new GLfloat[3 * numFloatsPerCell * nflat];
    GLuint* smokeIndexes = new GLuint[3 * 6 * nflat]; // 2 triangles

    // setting up positions for square faces
    int shift_x = gridSize.x / 2.f;
    int shift_z = gridSize.z / 2.f;

    // YZ plane
    for (unsigned int x = 0; x < gridCount.x; x++) {
        for (unsigned int y = 0; y < gridCount.y; y++) {
            for (unsigned int z = 0; z < gridCount.z; z++) {
                std::array<float, numFloatsPerCell> vertexes = {
                    x * sideLength - shift_x, y * sideLength, z * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    x * sideLength - shift_x, y * sideLength, (z + 1) * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    x * sideLength - shift_x, (y + 1) * sideLength, (z + 1) * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    x * sideLength - shift_x, (y + 1) * sideLength, z * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f
                };
                std::copy(vertexes.begin(), vertexes.end(),
                    smokeQuadsPositions + numFloatsPerCell * flatten(x, y, z));
            }
        }
    }

    // XZ plane
    for (unsigned int x = 0; x < gridCount.x; x++) {
        for (unsigned int y = 0; y < gridCount.x; y++) {
            for (unsigned int z = 0; z < gridCount.z; z++) {
                std::array<float, numFloatsPerCell> vertexes = {
                    x * sideLength - shift_x, y * sideLength, z * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    (x + 1) * sideLength - shift_x, y * sideLength, z * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    (x + 1) * sideLength - shift_x, y * sideLength, (z + 1) * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    x * sideLength - shift_x, y * sideLength, (z + 1) * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f
                };
                std::copy(vertexes.begin(), vertexes.end(),
                    smokeQuadsPositions + 1 * numFloatsPerCell * nflat + numFloatsPerCell * flatten(x, y, z));
            }
        }
    }

    // XY plane
    for (unsigned int x = 0; x < gridCount.x; x++) {
        for (unsigned int y = 0; y < gridCount.y; y++) {
            for (unsigned int z = 0; z < gridCount.z; z++) {
                std::array<float, numFloatsPerCell> vertexes = {
                    x * sideLength - shift_x, y * sideLength, z * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    (x + 1) * sideLength - shift_x, y * sideLength, z * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    (x + 1) * sideLength - shift_x, (y + 1) * sideLength, z * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f,
                    x * sideLength - shift_x, (y + 1) * sideLength, z * sideLength - shift_z,
                    0.f, 0.f, 0.f, 0.f
                };
                std::copy(vertexes.begin(), vertexes.end(),
                    smokeQuadsPositions + 2 * numFloatsPerCell * nflat + numFloatsPerCell * flatten(x, y, z));
            }
        }
    }

    for (GLuint i = 0; i < 3 * 6 * nflat; i += 6) {
        GLuint offset = (i / 6) * 4;
        smokeIndexes[i] = offset;
        smokeIndexes[i + 1] = offset + 1;
        smokeIndexes[i + 2] = offset + 2;
        smokeIndexes[i + 3] = offset;
        smokeIndexes[i + 4] = offset + 2;
        smokeIndexes[i + 5] = offset + 3;
    }

    glGenVertexArrays(1, &VAO_smoke);
    glGenBuffers(1, &VBO_smoke);
    glGenBuffers(1, &IBO_smoke);

    glBindVertexArray(VAO_smoke);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_smoke);
    glBufferData(GL_ARRAY_BUFFER, 3 * nflat * numFloatsPerCell * sizeof(GLfloat), smokeQuadsPositions, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO_smoke);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * 6 * nflat * sizeof(GLuint), smokeIndexes, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
}

void initVAO(int NUM_OF_BRANCHES) {
    
    /** Terrain */
    #ifdef BIG_SCENE
    float terrainSizeX = 110.f; // TODO: needs to be a function of terrain size from scene loading
    float terrainSizeZ = 25.f; // TODO: needs to be a function of terrain size from scene loading
    #else
    float terrainSizeX = 25.f; // TODO: needs to be a function of terrain size from scene loading
    float terrainSizeZ = 25.f; // TODO: needs to be a function of terrain size from scene loading
    #endif

    // Includes vertices + texture coords
    GLfloat vertices[] =
    {
        -terrainSizeX, 0.0f, -terrainSizeZ,  -1.0f, -1.0f, // bottom left
        -terrainSizeX, 0.0f, terrainSizeZ, -1.0f, 1.0f, // top left
        terrainSizeX, 0.f, terrainSizeZ, 1.0f, 1.0f, // top right
        terrainSizeX, 0.f, -terrainSizeZ, 1.0f, -1.0f // bottom right
    };

    GLushort indices[] =
    {
        0, 1, 2, 0, 2, 3
    };

    glGenVertexArrays(1, &VAO_terrain);
    glGenBuffers(1, &VBO_terrain);
    glGenBuffers(1, &IBO_terrain);

    glBindVertexArray(VAO_terrain);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_terrain);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO_terrain);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // texture
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); //texture wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); //texture filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int texWidth, texHeight, texChannels;
    unsigned char* pixels = stbi_load("shaders/forest-ground.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        //throw std::runtime_error("Failed to load texture image");
        std::cout << "Texture Failure" << std::endl;
    }
    else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    stbi_image_free(pixels);

    /** Branches */
    std::unique_ptr<GLfloat[]> branches{ new GLfloat[11 * NUM_OF_BRANCHES] };

    // This will be updated by the simulation in CUDA
    for (int i = 0; i < NUM_OF_BRANCHES; i++)
    {
        branches[11 * i + 0] = 0.0f;
        branches[11 * i + 1] = 0.0f;
        branches[11 * i + 2] = 0.0f;
        branches[11 * i + 3] = 0.0f;
        branches[11 * i + 4] = 0.0f;
        branches[11 * i + 5] = 0.0f;
        branches[11 * i + 6] = 0.0f;
        branches[11 * i + 7] = 0.0f;
        branches[11 * i + 8] = 0.0f;
        branches[11 * i + 9] = 0.0f;
        branches[11 * i + 10] = 0.0f;
    }


    glGenVertexArrays(1, &VAO_branches);
    glGenBuffers(1, &VBO_branches);

    glBindVertexArray(VAO_branches);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_branches);
    glBufferData(GL_ARRAY_BUFFER,  NUM_OF_BRANCHES * (11 * sizeof(GLfloat)), branches.get(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)(4 * sizeof(GLfloat)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)(8 * sizeof(GLfloat)));

    glBindVertexArray(0);
}

// GLFW Callbacks
void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
    GLuint location;
    if (key == GLFW_KEY_1 && action == GLFW_PRESS)
    {
        glUseProgram(program[PROG_branches]);
        glBindVertexArray(VAO_branches);
        // render module temperature
        renderModuleTemp = !renderModuleTemp;
        if ((location = glGetUniformLocation(program[PROG_branches], "u_renderTemp")) != -1) {
            glUniform1i(location, renderModuleTemp);
        }
    }
    if (key == GLFW_KEY_2 && action == GLFW_PRESS)
    {
        glUseProgram(program[PROG_branches]);
        glBindVertexArray(VAO_branches);
        // render temperature
        renderLeaves = !renderLeaves;
        if ((location = glGetUniformLocation(program[PROG_branches], "u_renderLeaves")) != -1) {
            glUniform1i(location, renderLeaves);
        }
    }
    if (key == GLFW_KEY_3 && action == GLFW_PRESS)
    {
        std::cout << "entered" << std::endl;
        glUseProgram(program[PROG_fluid]);
        glBindVertexArray(VAO_smoke);
        renderSmoke = !renderSmoke;
        if ((location = glGetUniformLocation(program[PROG_fluid], "u_renderSmoke")) != -1) {
            glUniform1i(location, renderSmoke);
        }
    }
  }

  void mouseButton(GLFWwindow* window, int button, int action, int mods) {
      camera.mouseButtonCallback(window, button, action, mods);
  }

  void mousePosition(GLFWwindow* window, double xpos, double ypos) {
      camera.mousePositionCallback(window, xpos, ypos, program);
  }
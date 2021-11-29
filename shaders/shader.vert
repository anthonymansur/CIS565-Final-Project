/*varying vec4 P;
varying vec3 N;
varying vec4 C;

void main(void) {
    P = gl_Vertex;
    N = gl_Normal;
    C = gl_Color;
    gl_Position = ftransform ();
}*/

#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 u_projMatrix;
  
out vec3 vertexColor;

void main()
{
    gl_Position = u_projMatrix * vec4(aPos, 1.0);
    vertexColor = vec3(0.24, 0.92, 0.42); 
}
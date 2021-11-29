/*varying vec4 P;
varying vec3 N;
varying vec4 C;

void main(void) {
    P = gl_Vertex;
    N = gl_Normal;
    C = gl_Color;
    gl_Position = ftransform ();
}*/

#version 330
layout (location = 0) in vec3 inPos;

uniform mat4 u_projMatrix;

out vec3 color;

void main() {
    gl_Position = u_projMatrix * vec4(inPos, 1.0);
    color = vec3(60, 230, 108); // green
}
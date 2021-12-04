#version 330
layout (location = 0) in vec3 pos;

uniform mat4 u_model;
uniform mat4 u_projMatrix;

void main() {
    gl_Position = u_projMatrix * u_model * vec4(pos, 1.0);
}
#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in vec4 col;

uniform mat4 u_model;
uniform mat4 u_projMatrix;

out vec4 f_col;

void main() {
    gl_Position = u_projMatrix * u_model * vec4(pos, 1.0);
    f_col = vec4(col.xyz, 0.5f);
}
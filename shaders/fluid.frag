#version 330

in vec4 f_col;
out vec4 outColor;

uniform bool u_renderSmoke;
uniform float u_time;

void main() {
    float transp = f_col.w;
    transp = cos(u_time) * 0.8 * transp;
    outColor = vec4(f_col.r, f_col.g, f_col.b, transp);
}
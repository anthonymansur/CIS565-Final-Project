#version 330

in vec4 f_col;
out vec4 outColor;

uniform bool u_renderSmoke;

void main() {
    if (u_renderSmoke) {
        outColor = f_col;
    } else {
        //outColor = vec4(1.f, 0.f, 0.f, 0.1f);
        outColor = f_col;
    }
    
}
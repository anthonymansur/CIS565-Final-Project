/*
varying vec4 P; // fragment-wise position
varying vec3 N; // fragment-wise normal
varying vec4 C; // fragment-wise normal

void main (void) {
    gl_FragColor = vec4 (0.0, 0.0, 0.0, 1.0);
    
    vec3 p = vec3 (gl_ModelViewMatrix * P);
    vec3 n = normalize (gl_NormalMatrix * N);
    vec3 l = normalize (lightPos - p);
    vec3 v = normalize (-p);
    
    // ---------- Code to change -------------
    vec4 color = C;
    // ----------------------------------------
    
    gl_FragColor += color;
}*/

// #version 330

// in vec3 color;
// out vec4 outColor;

// void main() {
//     outColor = vec4(color, 1.0f);
// }

#version 330 core
out vec4 FragColor;
  
in vec3 vertexColor; // the input variable from the vertex shader (same name and same type)  

void main()
{
    FragColor = vec4(vertexColor, 1.0f);
} 
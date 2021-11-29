#version 330
layout (location = 0) in vec4 in_v0;
layout (location = 1) in vec4 in_v1;

out vec4 out_v0;
out vec4 out_v1;

void main() {
   // TODO: implement

   gl_Position = vec4(in_v0.xyz, 1.f);
   out_v0 = in_v0;
   out_v1 = in_v1;
}
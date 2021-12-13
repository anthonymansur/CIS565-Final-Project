#version 330
layout (location = 0) in vec4 v0;
layout (location = 1) in vec4 v1;
layout (location = 2) in vec3 attrib;

out vec4 geo_v0;
out vec4 geo_v1;
out vec3 geo_attrib;

void main() {
   geo_v0 = v0;
   geo_v1 = v1;
   geo_attrib = attrib;

   gl_Position = vec4(v0.xyz, 1.f);
}
/*#version 330

in vec4 Position;
out vec4 vFragColorVs;

void main() {
	vFragColorVs = vec4(163, 249, 255, 255);
	gl_Position = Position;
}*/

#version 330
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 u_projMatrix;

out vec2 texCoord;
out vec3 color;

void main() {
	gl_Position = u_projMatrix * vec4(aPos, 1.0);
	color = aColor;
	texCoord = aTexCoord;
}
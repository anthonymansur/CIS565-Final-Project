#version 330

in vec2 texCoord;
in vec3 color;

out vec4 outColor;

uniform sampler2D ourTex;

void main() {
	//outColor = texture(ourTex, texCoord);
	outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
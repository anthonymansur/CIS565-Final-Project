#version 330

in vec2 texCoord;
in vec4 color;

out vec4 outColor;

uniform sampler2D ourTex;

void main() {
	outColor = texture(ourTex, texCoord);
	/*outColor.r = vFragColor.r;
	outColor.g = vFragColor.g;
	outColor.b = vFragColor.b;*/
}
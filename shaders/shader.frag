#version 330
uniform sampler2D texSampler;

in vec2 aTexCoord;

out vec4 outColor;

void main() {
    //outColor = vec4(.12, .31, .14, 1.f);
    outColor = texture(texSampler, aTexCoord);
}
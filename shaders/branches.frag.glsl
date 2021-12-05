#version 330

in float frag_attrib;
in float y_coord;
in vec3 aNormal;

out vec4 outColor;

void main() 
{
    vec4 color;
    if (frag_attrib > 0.f)
        color = vec4(.12, .31, .14, 1.f); // green
    else
        color = vec4(.38, .25, .13, 1.f); // dark brown

    vec3 ambient = vec3(0.4f) * color.rgb; //ambient 

    vec3 lightDir = vec3(0.2f, 1.0f, 0.3f); // diffuse
    float diff = dot(aNormal, lightDir);
    vec3 diffuse = vec3(0.5f) * diff * color.rgb;
    //float multiplier = (1.0 - y_coord) * 0.05 + y_coord * 0.05;

    outColor = vec4((ambient + diffuse), 1.0f);
}
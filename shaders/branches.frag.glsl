#version 330

in float frag_attrib;
in float v_coord;
in vec3 aNormal;

out vec4 outColor;

void main() 
{
    // if (frag_attrib > 0.f)
    //     outColor = vec4(.12, .31, .14, 1.f); // green
    // else

    vec4 color = vec4(.38, .25, .13, 1.f); // dark brown

    vec3 ambient = vec3(0.4f) * color.rgb; //ambient 

    vec3 lightDir = vec3(0.2f, 1.0f, 0.3f); // diffuse
    float diff = max(dot(aNormal, lightDir), 0.0);
    vec3 diffuse = vec3(0.5f) * diff * color.rgb;

    outColor = vec4((ambient + diffuse), 1.0f);

    //float multiplier = (0.5 - v_coord) * 0.05 + v_coord * 1.0;
    //outColor = /*multiplier **/ color;

}
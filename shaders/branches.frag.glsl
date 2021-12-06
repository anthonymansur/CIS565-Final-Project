#version 330

in vec2 frag_attrib;
in float y_coord;
in vec3 aNormal;

out vec4 outColor;

vec4 temp_color(float temp)
{
    if (temp < 0)
        return vec4(0.f, 1.f, 0.f, 1.f);
    else if (temp > 450)
        return vec4(1.f, 1.f, 0.f, 1.f);
    else
        return ((450 - temp) / 450) * vec4(0.f, 0.f, 1.f, 1.f) + 
        (1 - ((450 - temp) / 450)) * vec4(1.f, 0.f, 0.f, 1.f);
}

void main() 
{
    vec4 color;
    if (frag_attrib.x > 0.f)
        color = vec4(.12, .31, .14, 1.f); // green
    else
        color = temp_color(frag_attrib.y); // dark brown

    vec3 ambient = vec3(0.4f) * color.rgb; //ambient 

    vec3 lightDir = vec3(0.2f, 1.0f, 0.3f); // diffuse
    float diff = dot(aNormal, lightDir);
    vec3 diffuse = vec3(0.5f) * diff * color.rgb;
    //float multiplier = (1.0 - y_coord) * 0.05 + y_coord * 0.05;

    outColor = vec4((ambient + diffuse), 1.0f);
}
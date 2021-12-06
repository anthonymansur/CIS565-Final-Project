#version 330

in vec2 frag_attrib;
in vec3 frag_normal;
in float frag_height;

out vec4 outColor;

uniform bool u_renderTemp;

// function prototypes
vec4 temp_color(float temp);
float ambient(float height);

void main() 
{
    vec4 color;
    if (frag_attrib.x > 0.f)
        color = vec4(.12, .31, .14, 1.0); // green
    else
    {
        if (u_renderTemp)
        {
            outColor = temp_color(frag_attrib.y);
            return;
        }
            
        else
           color = vec4(.63, .58, .55, 1.0); 
    }

    vec3 ambient = ambient(frag_height) * color.rgb; // ambient 

    vec3 lightDir = vec3(0.2f, 1.0f, 0.3f); // diffuse
    float diff = dot(frag_normal, lightDir);
    vec3 diffuse = vec3(0.6f) * diff * color.rgb;

    outColor = vec4((ambient + diffuse), 1.0f);
}

// function implementations
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

float ambient(float height)
{
    return (1 - ((15 - frag_height) / 15)) * 0.3;
}
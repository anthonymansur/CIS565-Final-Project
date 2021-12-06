#version 330

in vec2 frag_attrib;
in vec3 frag_normal;
in float frag_height;

out vec4 outColor;

uniform bool u_renderTemp;

// function prototypes
vec4 temp_color(float temp);
float ambientFactor(float height);

vec4 leaf_color1 = vec4(.29, .42, .04, 1.f);
vec4 leaf_color2 = vec4(.75, .62, .13, 1.f);
vec4 leaf_color3 = vec4(.68, .33, .21, 1.f);

void main() 
{
    vec4 color;
    vec3 ambient;

    if (frag_attrib.x > 0.f)
    {
        // leaf render
        if (frag_attrib.y < 0.5)
            color = leaf_color1;
        else if (frag_attrib.y < 1.5)
            color = leaf_color2;
        else
            color = leaf_color3;
        ambient = 0.45 * color.rgb;
    }
    else
    {
        // branch render
        if (u_renderTemp)
        {
            outColor = temp_color(frag_attrib.y);
            return;
        }
            
        else
        {
            if (frag_attrib.y >= 150)
                color = 0.2 * vec4(.63, .58, .55, 1.0) + 0.8 * vec4(.81, .16, .01, 1.0);
            else
                color = vec4(.63, .58, .55, 1.0); 
        }

        ambient = ambientFactor(frag_height) * color.rgb; // ambient 
    }

    // lambertian
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
    {
        float heat = (1 - ((450 - temp) / 450));
        return vec4(heat, 0, 1-heat, 1);
    }
}

float ambientFactor(float height)
{
    return (1 - ((15 - frag_height) / 15)) * 0.4;
}
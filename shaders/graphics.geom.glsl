#version 330

uniform mat4 u_projMatrix;

layout(points) in;
layout(points) out;
layout(max_vertices = 4) out;

in vec4 vFragColorVs[];
out vec4 vFragColor;

//vec3 Position[4];

void main() {
	vec3 Position[];
	Position[0] = gl_in[0].gl_Position.xyz;
	Position[1] = gl_in[0].gl_Position.xyz;
	Position[2] = gl_in[0].gl_Position.xyz;
	Position[3] = gl_in[0].gl_Position.xyz;

	//vec3 Position = gl_in[0].gl_Position.xyz;

	vFragColor = vFragColorVs[0];
	gl_Position = u_projMatrix * vec4(Position[0], 1.0);
	EmitVertex();
	gl_Position = u_projMatrix * vec4(Position[1], 1.0);
	EmitVertex();
	gl_Position = u_projMatrix * vec4(Position[2], 1.0);
	EmitVertex();
	gl_Position = u_projMatrix * vec4(Position[3], 1.0);
	EmitVertex();

	EndPrimitive();
}
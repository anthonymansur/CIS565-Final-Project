#pragma once

class TruncatedCone
{
public:
	TruncatedCone(float r0, float r1, float l);
	TruncatedCone(const TruncatedCone& copy);

	float Area();
	float Volume();
	float Mass(float density);

private:
	float r0; // starting radius of truncated cone
	float r1; // ending radius of truncatd cone
	float l; // length of cone
};
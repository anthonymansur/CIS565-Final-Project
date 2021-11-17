#define _USE_MATH_DEFINES // Keep above math.h import

#include <math.h> 
#include "TruncatedCone.h"

TruncatedCone::TruncatedCone(float r0, float r1, float l) :
  r0(r0), r1(r1), l(l)
{}

TruncatedCone::TruncatedCone(const TruncatedCone& copy)
{
	this->r0 = copy.r0;
	this->r1 = copy.r1;
	this->l = copy.l;
}

float TruncatedCone::Area()
{
  return (float)M_PI * (r0 + r1) * sqrtf((r0 - r1)*(r0 - r1) + l*l);
}

float TruncatedCone::Volume()
{
  return (float)(M_PI / 3) * l * (r0*r0 + r0*r1 + r1*r1);
}

float TruncatedCone::Mass(float density)
{
  return density * Volume();
}
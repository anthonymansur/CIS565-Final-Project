#include "cylinder.h"
#include <iostream>

Cylinder::Cylinder(float base, float top, float ht, int sect, int stack) {
	this->baseRadius = base;
	this->topRadius = top;
	this->height = ht;
	this->sectorCount = sect;
	this->stackCount = stack;

	buildVertices();
}

void Cylinder::buildVertices() {
	// unit circle
    float sectorStep = 2 * PI / sectorCount;
    float sectorAngle;  // radians

    std::vector<float>().swap(circleVertices);
    for (int i = 0; i <= sectorCount; ++i) {
        sectorAngle = i * sectorStep;
        circleVertices.push_back(sin(sectorAngle)); // x
        circleVertices.push_back(0); // y
        circleVertices.push_back(cos(sectorAngle)); // z
    }

    clearArrays();
    
    float x, y, z, radius;
    float sectorAngle2;

    // normals - not sure if we need this currently
    float yAngle = atan2(baseRadius - topRadius, height);
    float x0 = 0;               // nx
    float y0 = sin(yAngle);     // ny
    float z0 = cos(yAngle);     // nz
    
    std::vector<float> sideNormals;
    for (int i = 0; i <= sectorCount; ++i) {
        sectorAngle2 = i * sectorStep;
        sideNormals.push_back(cos(sectorAngle2) * z0 - sin(sectorAngle2) * x0);   // nz
        sideNormals.push_back(sin(sectorAngle2) * z0 + cos(sectorAngle2) * x0);   // nx
        sideNormals.push_back(y0);  // ny
    }

    // scale unit circle
    for (int i = 0; i <= stackCount; ++i) {
        y = -(height * 0.5f) + (float)i / stackCount * height;      // vertex position y
        radius = baseRadius + (float)i / stackCount * (topRadius - baseRadius);     // lerp
        float t = 1.0f - (float)i / stackCount;   // top-to-bottom

        for (int j = 0, k = 0; j <= sectorCount; ++j, k += 3) {
            x = circleVertices[k];
            z = circleVertices[k + 2];
            vertices.push_back(x * radius); // position
            vertices.push_back(y);
            vertices.push_back(z * radius);
            normals.push_back(sideNormals[k]);  // normal
            normals.push_back(sideNormals[k + 1]);
            normals.push_back(sideNormals[k + 2]);
            texCoords.push_back((float)j / sectorCount); // tex coord
            texCoords.push_back(t);
        }
    }

    unsigned int baseVertexIndex = (unsigned int)vertices.size() / 3;

    y = -height * 0.5f;
    vertices.push_back(0); 
    vertices.push_back(y);
    vertices.push_back(0);
    normals.push_back(0); 
    normals.push_back(-1);
    normals.push_back(0);
    texCoords.push_back(0.5f);
    texCoords.push_back(0.5f);

    for (int i = 0, j = 0; i < sectorCount; ++i, j += 3) {
        x = circleVertices[j];
        z = circleVertices[j + 2];
        vertices.push_back(x * baseRadius);
        vertices.push_back(y);
        vertices.push_back(z * baseRadius);
        normals.push_back(0);
        normals.push_back(-1);
        normals.push_back(0);
        texCoords.push_back(-x * 0.5f + 0.5f);
        texCoords.push_back(-z * 0.5f + 0.5f);
    }

    unsigned int topVertexIndex = (unsigned int)vertices.size() / 3;

    y = height * 0.5f;
    vertices.push_back(0);
    vertices.push_back(y);
    vertices.push_back(0);
    normals.push_back(0);
    normals.push_back(1);
    normals.push_back(0);
    texCoords.push_back(0.5f);
    texCoords.push_back(0.5f);

    for (int i = 0, j = 0; i < sectorCount; ++i, j += 3) {
        x = circleVertices[j];
        z = circleVertices[j + 2];
        vertices.push_back(x * topRadius);
        vertices.push_back(y);
        vertices.push_back(z * topRadius);
        normals.push_back(0);
        normals.push_back(1);
        normals.push_back(0);
        texCoords.push_back(-x * 0.5f + 0.5f); //check this
        texCoords.push_back(z * 0.5f + 0.5f);
    }

    unsigned int k1, k2;
    for (int i = 0; i < stackCount; ++i) {
        k1 = i * (sectorCount + 1);     // bebinning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            // 2 trianles per sector
            indices.push_back(k1);
            indices.push_back(k1 + 1);
            indices.push_back(k2);
            indices.push_back(k2);
            indices.push_back(k1 + 1);
            indices.push_back(k2 + 1);

            // vertical lines for all stacks
            lineIndices.push_back(k1);
            lineIndices.push_back(k2);
            // horizontal lines
            lineIndices.push_back(k2);
            lineIndices.push_back(k2 + 1);
            if (i == 0) {
                lineIndices.push_back(k1);
                lineIndices.push_back(k1 + 1);
            }
        }
    }

    baseIndex = (unsigned int)indices.size();

    for (int i = 0, k = baseVertexIndex + 1; i < sectorCount; ++i, ++k) {
        if (i < (sectorCount - 1)) {
            indices.push_back(baseVertexIndex);
            indices.push_back(k + 1);
            indices.push_back(k);
        }
        else {
            indices.push_back(baseVertexIndex);
            indices.push_back(baseVertexIndex + 1);
            indices.push_back(k);
        }
    }

    topIndex = (unsigned int)indices.size();

    for (int i = 0, k = topVertexIndex + 1; i < sectorCount; ++i, ++k) {
        if (i < (sectorCount - 1)) {
            indices.push_back(topVertexIndex);
            indices.push_back(k);
            indices.push_back(k + 1);
        }
        else {
            indices.push_back(topVertexIndex);
            indices.push_back(k);
            indices.push_back(topVertexIndex + 1);
        }
    }

    // interleaved
    std::vector<float>().swap(interleavedVertices);

    std::size_t i, j;
    std::size_t count = vertices.size();
    for (i = 0, j = 0; i < count; i += 3, j += 2) {
        //interleavedVertices.push_back(vertices[i]);
        //interleavedVertices.push_back(vertices[i+1]);
        //interleavedVertices.push_back(vertices[i+2]);
        interleavedVertices.insert(interleavedVertices.end(), &vertices[i], &vertices[i] + 3);

        //interleavedVertices.push_back(normals[i]);
        //interleavedVertices.push_back(normals[i+1]);
        //interleavedVertices.push_back(normals[i+2]);
        interleavedVertices.insert(interleavedVertices.end(), &normals[i], &normals[i] + 3);

        //interleavedVertices.push_back(texCoords[j]);
        //interleavedVertices.push_back(texCoords[j+1]);
        interleavedVertices.insert(interleavedVertices.end(), &texCoords[j], &texCoords[j] + 2);
    }
}

void Cylinder::drawCylinder() {
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glVertexPointer(3, GL_FLOAT, stride, &interleavedVertices[0]);
    glNormalPointer(GL_FLOAT, stride, &interleavedVertices[3]);
    glTexCoordPointer(2, GL_FLOAT, stride, &interleavedVertices[6]);

    glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, indices.data());

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

void Cylinder::clearArrays() {
    std::vector<GLfloat>().swap(vertices);
    std::vector<float>().swap(normals);
    std::vector<float>().swap(texCoords);
    std::vector<GLuint>().swap(indices);
    std::vector<int>().swap(lineIndices);
}
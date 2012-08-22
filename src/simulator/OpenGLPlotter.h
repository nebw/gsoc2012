/**
 * Copyright (C) 2012 Benjamin Wild
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#pragma once

#include "Definitions.h"

#include "BasePlotter.h"

#include <glm/glm.hpp>

#include <GL/glew.h>
#include <GL/glfw.h>

#include <vector>

class OpenGLPlotter : public BasePlotter {
public:

    OpenGLPlotter(size_t numNeurons,
                  size_t index,
                  float dt);
    ~OpenGLPlotter();

    virtual void step(const state *curState,
                      const size_t t,
                      std::unique_ptr<float[]> const& sumFootprintAMPA,
                      std::unique_ptr<float[]> const& sumFootprintNMDA,
                      std::unique_ptr<float[]> const& sumFootprintGABAA);

    virtual void plot();

private:

    enum GraphType {
        LINES = 0,
        POINTS,
        LINESPOINTS
    };

    struct Point {
        GLfloat x;
        GLfloat y;
    };

    struct Color {
        GLfloat red;
        GLfloat green;
        GLfloat blue;
        GLfloat alpha;

        Color() {}

        Color(GLfloat r,
              GLfloat g,
              GLfloat b,
              GLfloat a)
            : red(r), green(g), blue(b), alpha(a) {}
    };

    struct GraphDesc {
        Color color;
        std::vector<Point> points;
    };

    struct PlotDesc {
        GraphType type;
        std::string title;
        size_t size;
        std::vector<GraphDesc> graphs;
    };

    GLuint _program;

    GLuint _vbo[3];
    GLint _attribute_coord2d;
    GLint _uniform_color;
    GLint _uniform_transform;
    int _border;
    int _ticksize;

    size_t _numNeurons;
    size_t _index;
    float _dt;

    std::vector<PlotDesc> _plots;

    std::vector<float> _V, _h, _n, _z, _sAMPA, _xNMDA, _sNMDA, _IApp;
    std::vector<float> _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA;
    // std::vector<float> _spikeTimes;
    // std::vector<size_t> _spikeNeuronIndicesX, _spikeNeuronIndicesY;
    // std::vector<bool> _spikeArr;

    glm::mat4 viewport_transform(float x,
                                 float y,
                                 float width,
                                 float height,
                                 float *pixel_x = 0,
                                 float *pixel_y = 0);
    void display();
    void initGL();
    void initGraphs();

    std::vector<Point> getPoints(std::vector<float> const& vec,
                                 size_t graphsize,
                                 float min,
                                 float max);
    std::pair<float, float> getMinMax(std::vector<std::vector<float> > input,
                                      const float scaleMin,
                                      const float scaleMax);

    static int _selPlot;
    static size_t _plotVecSize;
    static float _offset_x;
    static float _scale_x;

    static void GLFWCALL keyCallback(int key,
                                     int action);
};

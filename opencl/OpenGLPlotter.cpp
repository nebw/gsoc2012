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

#include "stdafx.h"

#include "OpenGLPlotter.h"
#include "utilShader.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <boost/filesystem.hpp>
#include <boost/scoped_array.hpp>
#include <boost/foreach.hpp>

OpenGLPlotter::OpenGLPlotter(unsigned int numNeurons, unsigned int index, float dt)
    : _numNeurons(numNeurons),
    _index(index),
    _dt(dt),
    _V(std::vector<float>()),
    _h(std::vector<float>()),
    _n(std::vector<float>()),
    _z(std::vector<float>()),
    _sAMPA(std::vector<float>()),
    _xNMDA(std::vector<float>()),
    _sNMDA(std::vector<float>()),
    _IApp(std::vector<float>()),
    _sumFootprintAMPA(std::vector<float>()),
    _sumFootprintNMDA(std::vector<float>()),
    _sumFootprintGABAA(std::vector<float>()),
    _plots(std::vector<PlotDesc>()),
    _border(10),
    _ticksize(10)
{
}

OpenGLPlotter::~OpenGLPlotter()
{
    glfwTerminate();
}


void OpenGLPlotter::step(const state *curState, const unsigned int t, std::unique_ptr<float[]> const& sumFootprintAMPA, std::unique_ptr<float[]> const& sumFootprintNMDA, std::unique_ptr<float[]> const& sumFootprintGABAA)
{
    _V.push_back(curState[_index].V);
    _h.push_back(curState[_index].h);
    _n.push_back(curState[_index].n);
    _z.push_back(curState[_index].z);
    _IApp.push_back(curState[_index].I_app);
    _sAMPA.push_back(curState[_index].s_AMPA);
    _xNMDA.push_back(curState[_index].x_NMDA);
    _sNMDA.push_back(curState[_index].s_NMDA);
    _sumFootprintAMPA.push_back(sumFootprintAMPA[_index]);
    _sumFootprintNMDA.push_back(sumFootprintNMDA[_index]);
    _sumFootprintGABAA.push_back(sumFootprintGABAA[_index]);
}

void OpenGLPlotter::plot()
{
    initGL();

    initGraphs();

    glfwDisable( GLFW_KEY_REPEAT );

    int running = GL_TRUE;
    while(running)
    {
        glfwSetWindowTitle(_plots[_selPlot].title.c_str());
        display();

        glfwSwapBuffers();

        running = !glfwGetKey( GLFW_KEY_ESC ) &&
                  glfwGetWindowParam( GLFW_OPENED );
    }
}

glm::mat4 OpenGLPlotter::viewport_transform( float x, float y, float width, float height, float *pixel_x /*= 0*/, float *pixel_y /*= 0*/ )
{
    // Map OpenGL coordinates (-1,-1) to window coordinates (x,y),
    // (1,1) to (x + width, y + height).

    // First, we need to know the real window size:
    int window_width, window_height;
    glfwGetWindowSize(&window_width, &window_height);

    // Calculate how to translate the x and y coordinates:
    float offset_x = (2.0f * x + (width - window_width)) / window_width;
    float offset_y = (2.0f * y + (height - window_height)) / window_height;

    // Calculate how to rescale the x and y coordinates:
    float scale_x = width / window_width;
    float scale_y = height / window_height;

    // Calculate size of pixels in OpenGL coordinates
    if(pixel_x)
        *pixel_x = 2.0f / width;
    if(pixel_y)
        *pixel_y = 2.0f / height;

    return glm::scale(glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, 0)), glm::vec3(scale_x, scale_y, 1));
}

void OpenGLPlotter::display()
{
    PlotDesc plotdesc = _plots[_selPlot];

    // Create a VBO for the border
    static const Point border_points[4] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    glBindBuffer(GL_ARRAY_BUFFER, _vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof border_points, border_points, GL_STATIC_DRAW);

    int window_width, window_height;
    glfwGetWindowSize(&window_width, &window_height);

    glUseProgram(_program);

    glClearColor(0.9f, 0.9f, 0.9f, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    /* ----------------------------------------------------------------*/
    /* Draw the graph */

    // Set our viewport, this will clip geometry
    glViewport(
        _border + _ticksize,
        _border + _ticksize,
        window_width - _border * 2 - _ticksize,
        window_height - _border * 2 - _ticksize
    );

    // Set the scissor rectangle,this will clip fragments
    glScissor(
        _border + _ticksize,
        _border + _ticksize,
        window_width - _border * 2 - _ticksize,
        window_height - _border * 2 - _ticksize
    );

    glEnable(GL_SCISSOR_TEST);

    // Set our coordinate transformation matrix
    glm::mat4 transform = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(_scale_x, 1, 1)), glm::vec3(_offset_x, 0, 0));
    glUniformMatrix4fv(_uniform_transform, 1, GL_FALSE, glm::value_ptr(transform));

    BOOST_FOREACH(GraphDesc const& graphdesc, plotdesc.graphs)
    {
        Color color = graphdesc.color;

        // Tell OpenGL to copy our array to the buffer object
        glBindBuffer(GL_ARRAY_BUFFER, _vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, plotdesc.size * sizeof(Point), &(graphdesc.points[0]), GL_STATIC_DRAW);

        // Set the color to red
        GLfloat red[4] = {color.red, color.green, color.blue, color.alpha};
        glUniform4fv(_uniform_color, 1, red);

        // Draw using the vertices in our vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, _vbo[0]);

        glEnableVertexAttribArray(_attribute_coord2d);
        glVertexAttribPointer(_attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_LINE_STRIP, 0, plotdesc.size);
    }

    // Stop clipping
    glViewport(0, 0, window_width, window_height);
    glDisable(GL_SCISSOR_TEST);

    /* ----------------------------------------------------------------*/
    /* Draw the _borders */

    float pixel_x, pixel_y;

    // Calculate a transformation matrix that gives us the same normalized device coordinates as above
    transform = viewport_transform(
        static_cast<float>(_border + _ticksize),
        static_cast<float>(_border + _ticksize),
        static_cast<float>(window_width - _border * 2 - _ticksize),
        static_cast<float>(window_height - _border * 2 - _ticksize),
        &pixel_x, &pixel_y
        );

    // Tell our vertex shader about it
    glUniformMatrix4fv(_uniform_transform, 1, GL_FALSE, glm::value_ptr(transform));

    // Set the color to black
    GLfloat black[4] = {0.1f, 0.1f, 0.1f, 1};
    glUniform4fv(_uniform_color, 1, black);

    // Draw a _border around our graph
    glBindBuffer(GL_ARRAY_BUFFER, _vbo[1]);
    glVertexAttribPointer(_attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_LINE_LOOP, 0, 4);

    /* ----------------------------------------------------------------*/
    /* Draw the y tick marks */

    Point ticks[42];

    for(int i = 0; i <= 20; i++) {
        float y = -1 + i * 0.1f;
        float tickscale = (i % 10) ? 0.5f : 1; 
        ticks[i * 2].x = -1;
        ticks[i * 2].y = y; 
        ticks[i * 2 + 1].x = -1 - _ticksize * tickscale * pixel_x;
        ticks[i * 2 + 1].y = y; 
    }

    glBindBuffer(GL_ARRAY_BUFFER, _vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof ticks, ticks, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(_attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_LINES, 0, 42);

    /* ----------------------------------------------------------------*/
    /* Draw the x tick marks */

    float tickspacing = 0.1f * powf(10, -floor(log10(_scale_x))); // desired space between ticks, in graph coordinates
    float left = -1.0f / _scale_x - _offset_x;                     // left edge, in graph coordinates
    float right = 1.0f / _scale_x - _offset_x;                     // right edge, in graph coordinates
    int left_i = static_cast<int>(ceil(left / tickspacing));                      // index of left tick, counted from the origin
    int right_i = static_cast<int>(floor(right / tickspacing));                   // index of right tick, counted from the origin
    float rem = left_i * tickspacing - left;                    // space between left edge of graph and the first tick

    float firsttick = -1.0f + rem * _scale_x;                     // first tick in device coordinates

    int nticks = right_i - left_i + 1;                          // number of ticks to show
    if(nticks > 21)
        nticks = 21; // should not happen

    for(int i = 0; i < nticks; i++) {
        float x = firsttick + i * tickspacing * _scale_x;
        float tickscale = ((i + left_i) % 10) ? 0.5f : 1; 
        ticks[i * 2].x = x; 
        ticks[i * 2].y = -1;
        ticks[i * 2 + 1].x = x; 
        ticks[i * 2 + 1].y = -1 - _ticksize * tickscale * pixel_y;
    }

    glBufferData(GL_ARRAY_BUFFER, sizeof ticks, ticks, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(_attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_LINES, 0, nticks * 2);

    // And we are done.

    glDisableVertexAttribArray(_attribute_coord2d);
}

void OpenGLPlotter::initGL()
{
    auto pathVertexShader = boost::filesystem::path(CL_SOURCE_DIR);
    pathVertexShader /= "/vertexShader.glsl";
    auto pathFragmentShader = boost::filesystem::path(CL_SOURCE_DIR);
    pathFragmentShader /= "/fragmentShader.glsl";

    if(!glfwInit())
    {
        throw;
    };

    if(!glfwOpenWindow(800, 600, 0, 0, 0, 0, 0, 0, GLFW_WINDOW))
    {
        throw;
    };

    // enable vsync
    glfwSwapInterval(1);

    GLenum glew_status = glewInit();
    if (GLEW_OK != glew_status) {
        //fprintf(stderr, "Error: %s\n", glewGetErrorString(glew_status));
        throw glew_status;
    }

    if (!GLEW_VERSION_2_0) {
        //fprintf(stderr, "No support for OpenGL 2.0 found\n");
        throw;
    }

    GLint link_ok = GL_FALSE;

    GLuint vs, fs;
    if((vs = create_shader(pathVertexShader.string().c_str(), GL_VERTEX_SHADER)) == 0) 
        throw;
    if((fs = create_shader(pathFragmentShader.string().c_str(), GL_FRAGMENT_SHADER)) == 0)
        throw;

    _program = glCreateProgram();
    glAttachShader(_program, vs);
    glAttachShader(_program, fs);
    glLinkProgram(_program);
    glGetProgramiv(_program, GL_LINK_STATUS, &link_ok);
    if (!link_ok) {
        throw;
    }

    const char* attribute_name;
    attribute_name = "coord2d";
    _attribute_coord2d = glGetAttribLocation(_program, attribute_name);
    if (_attribute_coord2d == -1) {
        throw;
    }

    const char* uniform_name;
    uniform_name = "transform";
    _uniform_transform = glGetUniformLocation(_program, uniform_name);
    if (_uniform_transform == -1) {
        throw;
    }

    uniform_name = "color";
    _uniform_color = glGetUniformLocation(_program, uniform_name);
    if (_uniform_color == -1) {
        throw;
    }

    glGenBuffers(3, _vbo);

    OpenGLPlotter::_selPlot = 0;
    OpenGLPlotter::_offset_x = 0.0f;
    OpenGLPlotter::_scale_x = 0.1f;
    glfwEnable( GLFW_KEY_REPEAT );    glfwSetKeyCallback(keyCallback);
}

void OpenGLPlotter::initGraphs()
{
    size_t graphsize = _V.size();
    assert(_sAMPA.size() == graphsize);
    assert(_sNMDA.size() == graphsize);
    assert(_xNMDA.size() == graphsize);

    PlotDesc plotdesc;
    plotdesc.size = graphsize;
    plotdesc.title = "Membrane potential";
    plotdesc.type = LINESPOINTS;
    auto& minMax = std::minmax_element(_V.begin(), _V.end());
    float range = *minMax.second - *minMax.first;
    float min = *minMax.first - abs(range * 0.1f);
    float max = *minMax.second + abs(range * 0.1f);
    GraphDesc graphdesc; 
    std::vector<Point> points = std::move(getPoints(_V, graphsize, min, max));
    graphdesc.color = Color(1, 0, 0, 1);
    graphdesc.points = points;
    plotdesc.graphs.push_back(std::move(graphdesc));
    _plots.push_back(std::move(plotdesc));

    plotdesc = PlotDesc();
    plotdesc.size = graphsize;
    plotdesc.title = "TODO";
    plotdesc.type = LINESPOINTS;
    std::vector<std::vector<float>> input;
    input.push_back(_sAMPA);
    input.push_back(_sNMDA);
    input.push_back(_xNMDA);
    auto& minMaxInput = getMinMax(input, 0.0f, 0.0f);
    min = minMaxInput.first;
    max = minMaxInput.second;
    BOOST_FOREACH(std::vector<float> const& vals, input)
    {
        graphdesc = GraphDesc(); 
        points = std::move(getPoints(vals, graphsize, min, max));
        graphdesc.points = points;
        plotdesc.graphs.push_back(std::move(graphdesc));
    }
    plotdesc.graphs[0].color = Color(1, 0, 0, 1);
    plotdesc.graphs[1].color = Color(0, 1, 0, 1);
    plotdesc.graphs[2].color = Color(0, 0, 1, 1);
    _plots.push_back(std::move(plotdesc));

    plotdesc = PlotDesc();
    plotdesc.size = graphsize;
    plotdesc.title = "TODO";
    plotdesc.type = LINESPOINTS;
    input.clear();
    input.push_back(_h);
    input.push_back(_n);
    input.push_back(_z);
    minMaxInput = getMinMax(input, 0.0f, 0.0f);
    min = minMaxInput.first;
    max = minMaxInput.second;
    BOOST_FOREACH(std::vector<float> const& vals, input)
    {
        graphdesc = GraphDesc(); 
        points = std::move(getPoints(vals, graphsize, min, max));
        graphdesc.points = points;
        plotdesc.graphs.push_back(std::move(graphdesc));
    }
    plotdesc.graphs[0].color = Color(1, 0, 0, 1);
    plotdesc.graphs[1].color = Color(0, 1, 0, 1);
    plotdesc.graphs[2].color = Color(0, 0, 1, 1);
    _plots.push_back(std::move(plotdesc));

    plotdesc = PlotDesc();
    plotdesc.size = graphsize;
    plotdesc.title = "Synaptic fields";
    plotdesc.type = LINESPOINTS;
    input.clear();
    input.push_back(_sumFootprintAMPA);
    input.push_back(_sumFootprintNMDA);
    input.push_back(_sumFootprintGABAA);
    minMaxInput = getMinMax(input, 0.0f, 0.0f);
    min = minMaxInput.first;
    max = minMaxInput.second;
    BOOST_FOREACH(std::vector<float> const& vals, input)
    {
        graphdesc = GraphDesc(); 
        points = std::move(getPoints(vals, graphsize, min, max));
        graphdesc.points = points;
        plotdesc.graphs.push_back(std::move(graphdesc));
    }
    plotdesc.graphs[0].color = Color(1, 0, 0, 1);
    plotdesc.graphs[1].color = Color(0, 1, 0, 1);
    plotdesc.graphs[2].color = Color(0, 0, 1, 1);
    _plots.push_back(std::move(plotdesc));

    OpenGLPlotter::_plotVecSize = _plots.size();
}

void GLFWCALL OpenGLPlotter::keyCallback( int key, int action )
{
    switch (key)
    {
    case GLFW_KEY_PAGEUP:
        if(action == GLFW_PRESS)
            OpenGLPlotter::_selPlot = (OpenGLPlotter::_selPlot - 1) % OpenGLPlotter::_plotVecSize; 
    	break;
    case GLFW_KEY_PAGEDOWN:
        if(action == GLFW_PRESS)
            OpenGLPlotter::_selPlot = (OpenGLPlotter::_selPlot + 1) % OpenGLPlotter::_plotVecSize; 
        break;
    case GLFW_KEY_UP:
        OpenGLPlotter::_scale_x *= 1.5f;
        break;
    case GLFW_KEY_DOWN:
        OpenGLPlotter::_scale_x /= 1.5f;
        break;
    case GLFW_KEY_RIGHT:
        OpenGLPlotter::_offset_x -= 0.3f;
        break;
    case GLFW_KEY_LEFT:
        OpenGLPlotter::_offset_x += 0.3f;
        break;
    }
}

std::vector<OpenGLPlotter::Point> OpenGLPlotter::getPoints( std::vector<float> const& vec, size_t graphsize, float min, float max )
{
    std::vector<Point> points(graphsize);
    for(size_t i = 0; i < graphsize; i++) {
        float x = (i - 1000.0f) / 100.0f;
        points[i].x = x;
        points[i].y = 2 * (vec[i] - min) / (max - min) - 1;
    }
    return points;
}

std::pair<float, float> OpenGLPlotter::getMinMax( std::vector<std::vector<float>> input, const float scaleMin, const float scaleMax )
{
    float min = 0;
    float max = 0;
    float range;
    bool first = true;

    BOOST_FOREACH(auto const& vec, input)
    {
        auto minMax = std::minmax_element(vec.begin(), vec.end());
        if (first || *minMax.first < min)
            min = *minMax.first;
        if (first || *minMax.second > max)
            max = *minMax.second;
        if (first)
            first = false;
    }

    range = max - min;
    min = min - (range * scaleMin);
    max = max + (range * scaleMax);

    return std::make_pair(min, max);
}

float OpenGLPlotter::_scale_x;

float OpenGLPlotter::_offset_x;

size_t OpenGLPlotter::_plotVecSize;

int OpenGLPlotter::_selPlot;

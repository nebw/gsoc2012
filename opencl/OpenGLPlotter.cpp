#include "stdafx.h"

#include "OpenGLPlotter.h"
#include "utilShader.h"

#include <GL/glew.h>
#include <GL/glfw.h>

#include <boost/filesystem.hpp>
#include <boost/scoped_array.hpp>
#include <boost/foreach.hpp>

OpenGLPlotter::OpenGLPlotter(unsigned int numNeurons, unsigned int index, float dt)
    : _numNeurons(numNeurons),
    _index(index),
    _dt(dt),
    _graphsize(1000),
    _V(std::deque<float>()),
    _sAMPA(std::deque<float>()),
    _xNMDA(std::deque<float>()),
    _sNMDA(std::deque<float>()),
    _graph(std::unique_ptr<point[]>(new point[1000]))
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

    //enable vsync
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
        //fprintf(stderr, "glLinkProgram:");
        //print_log(program);
        //return 0;
        throw;
    }

    const char* attribute_name;
    attribute_name = "coord2d";
    _attribute_coord2d = glGetAttribLocation(_program, attribute_name);
    if (_attribute_coord2d == -1) {
        throw;
        /*fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
        return 0;*/
    }

    const char* uniform_name;
    uniform_name = "color";
    _uniform_color = glGetUniformLocation(_program, uniform_name);
    if (_uniform_color == -1) {
        fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
        throw;
    }

    glViewport(0, 0, _graphsize, 500);

//    /* Enable blending */
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Create the vertex buffer object
    glGenBuffers(1, &_vbo);

    //for(unsigned int i = 0; i < _graphsize - 1; ++i)
    //{
    //    _V.push_back(-100);
    //    _sAMPA.push_back(-100);
    //    _sNMDA.push_back(-100);
    //    _xNMDA.push_back(-100);
    //}
}

OpenGLPlotter::~OpenGLPlotter()
{
    //glfwTerminate();
}


void OpenGLPlotter::step(const state *curState, const unsigned int t, std::unique_ptr<float[]> const& sumFootprintAMPA, std::unique_ptr<float[]> const& sumFootprintNMDA, std::unique_ptr<float[]> const& sumFootprintGABAA)
{
    static const unsigned int interval = 10;

    _V.push_back(curState[_index].V);
    _sAMPA.push_back(curState[_index].s_AMPA);
    _sNMDA.push_back(curState[_index].s_NMDA);
    _xNMDA.push_back(curState[_index].x_NMDA);
    int size = _V.size();

    if(t % interval == 0)
    {
        glUseProgram(_program);

        glClearColor(255.0, 255.0, 255.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        /* Draw using the vertices in our vertex buffer object */
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);

        for(int i=0; i < size; ++i)
        {
            _graph[i].x = (i - (static_cast<int>(_graphsize)/2)) / (size/2.0);
            _graph[i].y = _sAMPA[size-i-1] - 1;
        }

        // Tell OpenGL to copy our array to the buffer object
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(point), _graph.get(), GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(_attribute_coord2d);
        glVertexAttribPointer(
            _attribute_coord2d,   // attribute
            2,                   // number of elements per vertex, here (x,y)
            GL_FLOAT,            // the type of each element
            GL_FALSE,            // take our values as-is
            0,                   // no space between values
            0                    // use the vertex buffer object
            );

        glUniform4f(_uniform_color, 0, 0, 1, 1);
        glDrawArrays(GL_LINE_STRIP, 0, size);

        for(int i=0; i < size; ++i)
        {
            _graph[i].x = (i - (static_cast<int>(_graphsize)/2)) / (size/2.0);
            _graph[i].y = _sNMDA[size-i-1] - 1;
        }

        // Tell OpenGL to copy our array to the buffer object
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(point), _graph.get(), GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(_attribute_coord2d);
        glVertexAttribPointer(
            _attribute_coord2d,   // attribute
            2,                   // number of elements per vertex, here (x,y)
            GL_FLOAT,            // the type of each element
            GL_FALSE,            // take our values as-is
            0,                   // no space between values
            0                    // use the vertex buffer object
            );

        glUniform4f(_uniform_color, 0, 1, 0, 1);
        glDrawArrays(GL_LINE_STRIP, 0, size);

        for(int i=0; i < size; ++i)
        {
            _graph[i].x = (i - (static_cast<int>(_graphsize)/2)) / (size/2.0);
            _graph[i].y = _xNMDA[size-i-1] - 1;
        }

        // Tell OpenGL to copy our array to the buffer object
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(point), _graph.get(), GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(_attribute_coord2d);
        glVertexAttribPointer(
            _attribute_coord2d,   // attribute
            2,                   // number of elements per vertex, here (x,y)
            GL_FLOAT,            // the type of each element
            GL_FALSE,            // take our values as-is
            0,                   // no space between values
            0                    // use the vertex buffer object
            );

        glUniform4f(_uniform_color, 0, 0.5, 0.5, 1);
        glDrawArrays(GL_LINE_STRIP, 0, size);

        for(int i=0; i < size; ++i)
        {
            _graph[i].x = (i - (static_cast<int>(_graphsize)/2)) / (size/2.0);
            _graph[i].y = _V[size-i-1] / 80.0 + 0.2;
        }

        // Tell OpenGL to copy our array to the buffer object
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(point), _graph.get(), GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(_attribute_coord2d);
        glVertexAttribPointer(
            _attribute_coord2d,   // attribute
            2,                   // number of elements per vertex, here (x,y)
            GL_FLOAT,            // the type of each element
            GL_FALSE,            // take our values as-is
            0,                   // no space between values
            0                    // use the vertex buffer object
            );

        //glUniform1f(uniform_sprite, 0);
        glUniform4f(_uniform_color, 1, 0, 0, 1);
        glDrawArrays(GL_LINE_STRIP, 0, size);

        /* Stop using the vertex buffer object */
        glDisableVertexAttribArray(_attribute_coord2d);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glfwSwapBuffers();
        if(size >= _graphsize)
        {
            for(unsigned int j = 0; j < interval; ++j)
            {
                _V.pop_front();
                _sAMPA.pop_front();
                _sNMDA.pop_front();
                _xNMDA.pop_front();
            }
        }
    }
}

void OpenGLPlotter::plot()
{
    glfwTerminate();
    //getchar();
}

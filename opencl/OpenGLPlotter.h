#pragma once

#include "Definitions.h"
#include "BasePlotter.h"

#include <deque>

class OpenGLPlotter : public BasePlotter
{
public:
    OpenGLPlotter(unsigned int numNeurons, unsigned int index, float dt);
    ~OpenGLPlotter();

    virtual void step(const state *curState, 
                      const unsigned int t, 
                      std::unique_ptr<float[]> const& sumFootprintAMPA, 
                      std::unique_ptr<float[]> const& sumFootprintNMDA, 
                      std::unique_ptr<float[]> const& sumFootprintGABAA) override;

    virtual void plot() override;

private:
    struct point {
        GLfloat x;
        GLfloat y;
    };

    const unsigned int _graphsize;

    GLuint _program;

    GLuint _vbo;
    GLint _attribute_coord2d;
    GLint _uniform_color;

    std::unique_ptr<point[]> _graph;

    unsigned int _numNeurons;
    unsigned int _index;
    float _dt;

    std::deque<float> _V, _sAMPA, _xNMDA, _sNMDA;



    //std::vector<float> _V, _h, _n, _z, _sAMPA, _xNMDA, _sNMDA, _IApp;
    //std::vector<float> _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA;
    //std::vector<float> _spikeTimes, _spikeNeuronIndices;
    //std::vector<bool> _spikeArr;
};
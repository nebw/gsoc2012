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

#include "GnuPlotPlotter.h"
#include "util.h"

#include "gnuplot_i/gnuplot_i.h"

#include <boost/foreach.hpp>

GnuPlotPlotter::GnuPlotPlotter(
    const unsigned int nX,
    const unsigned int nY,
    const unsigned int nZ,
    unsigned int index, 
    float dt)
    : _nX(nX),
      _nY(nY),
      _nZ(nZ),
      _numNeurons(nX * nY * nZ),
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
      _spikeTimes(std::vector<float>()),
      _spikeNeuronIndicesX(std::vector<unsigned int>()),
      _spikeNeuronIndicesY(std::vector<unsigned int>()),
      _spikeArr(std::vector<bool>(nX * nY * nZ))
{
}

void GnuPlotPlotter::step(const state *curState, const unsigned int t, std::unique_ptr<float[]> const& sumFootprintAMPA, std::unique_ptr<float[]> const& sumFootprintNMDA, std::unique_ptr<float[]> const& sumFootprintGABAA)
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

    unsigned int y = 0;
    for (unsigned int i = 0; i < _numNeurons; ++i)
    {
        if ((i >= _nX) && (i % _nX == 0))
        {
            ++y;
        }
        if ((curState[i].V >= 20) && (_spikeArr[i] == false))
        {
            _spikeTimes.push_back(t * _dt);
            _spikeNeuronIndicesX.push_back(i % _nX);
            _spikeNeuronIndicesY.push_back(y);
            _spikeArr[i] = true;
        } else if ((curState[i].V < 20) && (_spikeArr[i] == true))
        {
            _spikeArr[i] = false;
        }
    }
}

void GnuPlotPlotter::plot()
{
    unsigned int timesteps = _V.size();

    Gnuplot plot_V_Iapp_e;

    plot_V_Iapp_e.set_style("lines");
    plot_V_Iapp_e.set_title("Excitatory neuron");
    plot_V_Iapp_e.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _V, "V");
    plot_V_Iapp_e.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _IApp, "I_app");
    Gnuplot plot_sumFootprints;
    plot_sumFootprints.set_style("lines");
    plot_sumFootprints.set_title("Excitatory neuron");
    plot_sumFootprints.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _sumFootprintAMPA, "AMPA");
    plot_sumFootprints.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _sumFootprintNMDA, "NMDA");
    plot_sumFootprints.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _sumFootprintGABAA, "GABAA");
    Gnuplot plot_hnz_e;
    plot_hnz_e.set_style("lines");
    plot_hnz_e.set_title("Excitatory neuron");
    plot_hnz_e.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _h, "h");
    plot_hnz_e.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _n, "n");
    plot_hnz_e.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _z, "z");
    Gnuplot plot_Syn_e;
    plot_Syn_e.set_style("lines");
    plot_Syn_e.set_title("Excitatory neuron");
    plot_Syn_e.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _sAMPA, "s_AMPA");
    plot_Syn_e.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _xNMDA, "x_NMDA");
    plot_Syn_e.plot_xy(
        linSpaceVec<float>(0, (float)timesteps, timesteps),
        _sNMDA, "s_NMDA");
    Gnuplot plot_Spikes;
    plot_Spikes.set_title("Spikes");
    plot_Spikes.set_style("points");
    if(_nY == 1 && _nZ == 1)
    {
        plot_Spikes.set_xrange(0, timesteps * _dt);
        plot_Spikes.set_yrange(0, _numNeurons - 1);
        plot_Spikes.plot_xy(
            _spikeTimes, _spikeNeuronIndicesX, "Excitatory Spikes");
        getchar();
    } else if(_nZ == 1)
    {
        plot_Spikes.set_xlabel("Time");
        plot_Spikes.set_ylabel("X");
        plot_Spikes.set_zlabel("Y");
        plot_Spikes.set_xrange(0, timesteps * _dt);
        plot_Spikes.set_yrange(0, _nX - 1);
        plot_Spikes.set_zrange(0, _nY - 1);
        plot_Spikes.plot_xyz(
            _spikeTimes, _spikeNeuronIndicesX, _spikeNeuronIndicesY, "Excitatory Spikes");
        getchar();
    } else
    {
        getchar();
    }
}

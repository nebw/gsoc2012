#include "stdafx.h"

#include "Plotter.h"
#include "util.h"

#include "gnuplot_i/gnuplot_i.hpp"

#include <boost/foreach.hpp>

Plotter::Plotter(unsigned int numNeurons,
                 unsigned int index,
                 float dt)
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
      _spikeTimes(std::vector<float>()),
      _spikeNeuronIndices(std::vector<float>()),
      _spikeArr(std::vector<bool>(numNeurons))
{
    std::vector<float> _V, _h, _n, _z, _sAMPA, _xNMDA, _sNMDA, IApp;
    std::vector<float> _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA;
    std::vector<float> _spikeTimes, _spikeNeuronIndices;
    std::vector<bool> spikeArr;

    BOOST_FOREACH(bool val, spikeArr)
    {
        val = false;
    }
}

void Plotter::step(const state *curState, const unsigned int t, std::unique_ptr<float[]> const& sumFootprintAMPA, std::unique_ptr<float[]> const& sumFootprintNMDA, std::unique_ptr<float[]> const& sumFootprintGABAA)
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

    for (unsigned int i = 0; i < _numNeurons; ++i)
    {
        if ((curState[i].V >= 20) && (_spikeArr[i] == false))
        {
            _spikeTimes.push_back(t * _dt);
            _spikeNeuronIndices.push_back(i);
            _spikeArr[i] = true;
        } else if ((curState[i].V < 20) && (_spikeArr[i] == true))
        {
            _spikeArr[i] = false;
        }
    }
}

void Plotter::plot()
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
    plot_Spikes.set_xrange(0, timesteps * _dt);
    plot_Spikes.set_yrange(0, _numNeurons - 1);
    plot_Spikes.plot_xy(
        _spikeTimes, _spikeNeuronIndices, "Excitatory Spikes");

    getchar();
}

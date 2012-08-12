#include "Definitions.h"
#include "Simulator.h"
#include "util.h"

#include "cpplog/cpplog.hpp"

#include <numeric>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void measureTimes(Logger const& logger, state const& state0, boost::filesystem::path const& path)
{
    const int start = 0;
    const int powers = 10;

    std::vector<unsigned int> numNeurons;
    std::vector<double> avgTimesCalculations;
    std::vector<double> avgTimesFFTW;
    std::vector<double> avgTimesClFFT;

    for (int i = start; i < powers; ++i)
    {
        auto neurons = static_cast<const unsigned int>(pow(2.f, i));
        numNeurons.push_back(neurons);

        Simulator sim(
            neurons,
            500,
            0.1f,
            state0,
            Simulator::NO_PLOT,
            Simulator::MEASURE,
            Simulator::FFTW,
            Simulator::CLFFT,
            path,
            logger);

        sim.simulate();

        auto timesCalculations = sim.getTimesCalculations();
        auto timesFFTW = sim.getTimesFFTW();
        auto timesClFFT = sim.getTimesClFFT();

        avgTimesCalculations.push_back(
            std::accumulate(
                timesCalculations.begin(),
                timesCalculations.end(), 0.0)
            / timesCalculations.size());

        avgTimesFFTW.push_back(
            std::accumulate(
                timesFFTW.begin(),
                timesFFTW.end(), 0.0)
            / timesFFTW.size());

        avgTimesClFFT.push_back(
            std::accumulate(
                timesClFFT.begin(),
                timesClFFT.end(), 0.0)
            / timesClFFT.size());
    }

    for (int i = 0; i < powers - start; ++i)
    {
        std::cout << static_cast<const unsigned int>(pow(2.f, (int)(i + start))) << "\t" << avgTimesCalculations[i] << "\t" << avgTimesFFTW[i] << "\t" << avgTimesClFFT[i] << std::endl;
    }
}

int main(int ac, char **av)
{
    float V0, h0, n0, z0, sAMPA0, sNMDA0, xNMDA0, sGABAA0, IApp0, dt;
    unsigned int numNeurons, timesteps;
    std::string plot, measure, fftw, clfft;

    po::options_description desc("Allowed options");

    desc.add_options()
        ("help", "produce help message")
        ("V", po::value<float>(&V0)->default_value(-70.0), "initial value for membrane potential")
        ("h", po::value<float>(&h0)->default_value(1.0), "initial value for h")
        ("n", po::value<float>(&n0)->default_value(0.0), "initial value for n")
        ("z", po::value<float>(&z0)->default_value(0.0), "initial value for z")
        ("sAMPA", po::value<float>(&sAMPA0)->default_value(0.0), "initial value for sAMPA")
        ("sNMDA", po::value<float>(&sNMDA0)->default_value(0.0), "initial value for sNMDA")
        ("xNMDA", po::value<float>(&xNMDA0)->default_value(0.0), "initial value for xNMDA")
        ("sGABAA", po::value<float>(&sGABAA0)->default_value(0.0), "initial value for xNMDA")
        ("IApp", po::value<float>(&IApp0)->default_value(1.0), "initial value for IApp")
        ("dt", po::value<float>(&dt)->default_value(0.1f), "length of one timestep")
        ("timesteps", po::value<unsigned int>(&timesteps)->default_value(500), "number of timesteps")
        ("neurons", po::value<unsigned int>(&numNeurons)->default_value(1024), "number of neurons in network")
        ("plot", po::value<std::string>(&plot)->default_value("true"), "plot results")
        ("measure", po::value<std::string>(&measure)->default_value("true"), "measure execution time")
        ("fftw", po::value<std::string>(&fftw)->default_value("false"), "compute synaptic fields using fftw")
        ("clfft", po::value<std::string>(&clfft)->default_value("true"), "compute synaptic fields using clFFT")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(1);
    }

    auto logger = std::make_shared<cpplog::StdErrLogger>();
    state state0;
    state0.V = V0;
    state0.h = h0;
    state0.n = n0;
    state0.z = z0;
    state0.s_AMPA = sAMPA0;
    state0.s_NMDA = sNMDA0;
    state0.x_NMDA = xNMDA0;
    state0.s_GABAA = sGABAA0;
    state0.I_app = IApp0;

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    plot = "true";
    fftw = "true";
    clfft = "true";

    Simulator sim(
        numNeurons,
        timesteps,
        dt,
        state0,
        stringToBool(plot) ? Simulator::PLOT : Simulator::NO_PLOT,
        stringToBool(measure) ? Simulator::MEASURE : Simulator::NO_MEASURE,
        stringToBool(fftw) ? Simulator::FFTW : Simulator::NO_FFTW,
        stringToBool(clfft) ? Simulator::CLFFT : Simulator::NO_CLFFT,
        path,
        logger);

    sim.simulate();

    // //measureTimes(logger, state0, path);

    exit(0);
}

#include "Definitions.h"
#include "Simulator.h"
#include "util.h"

#include "cpplog/cpplog.hpp"

#include <numeric>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>

namespace po = boost::program_options;

void measureTimes(Logger const& logger, state const& state0, boost::filesystem::path const& path) 
{
    const unsigned int start = 0;
    const unsigned int powers = 10;

    std::vector<const unsigned int> numNeurons;
    std::vector<const double> avgTimesCalculations;
    std::vector<const double> avgTimesFFTW;
    std::vector<const double> avgTimesClFFT;

    for(int i = start; i < powers; ++i)
    {
        auto neurons = static_cast<const unsigned int>(pow(2.f, i));
        numNeurons.push_back(neurons);

        Simulator sim = Simulator(
            neurons,
            500,
            1,
            1,
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

    for(int i = 0; i < powers - start; ++i)
    {
        std::cout << static_cast<const unsigned int>(pow(2.f, (int)(i + start))) << "\t" << avgTimesCalculations[i] << "\t" << avgTimesFFTW[i] << "\t" << avgTimesClFFT[i] << std::endl;
    }
}

int finish()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    _CrtDumpMemoryLeaks();
#endif

    return(0);
}

int main(int ac, char **av)
{
    float V0, h0, n0, z0, sAMPA0, sNMDA0, xNMDA0, sGABAA0, IApp0, dt;
    unsigned int nX, nY, nZ, timesteps;
    std::string plotStr, measureStr, fftwStr, clfftStr;

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
        ("nX", po::value<unsigned int>(&nX)->default_value(1024), "number of neurons in network (X axis)")
        ("nY", po::value<unsigned int>(&nY)->default_value(1), "number of neurons in network (Y axis)")
        ("nZ", po::value<unsigned int>(&nZ)->default_value(1), "number of neurons in network (Z axis)")
        ("plot", po::value<std::string>(&plotStr)->default_value("false"), "plot results (false | gnuplot | opengl)")
        ("measure", po::value<std::string>(&measureStr)->default_value("true"), "measure execution time")
        ("fftw", po::value<std::string>(&fftwStr)->default_value("false"), "compute synaptic fields using fftw")
        ("clfft", po::value<std::string>(&clfftStr)->default_value("true"), "compute synaptic fields using clFFT")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(ac, av, desc), vm);
    } catch (po::error& err) {
        std::cout << desc << std::endl;
        exit(1);
    }

    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(1);
    }

    if(stringToBool(clfftStr) && !isPowerOfTwo(nX * nY * nZ))
    {
        std::cout << "Error: numNeurons (nX * nY * nZ) must be radix 2" << std::endl;
        exit(1);
    }

    auto logger = make_shared<cpplog::StdErrLogger>();
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

    Simulator::Plot plot;
    if(boost::iequals(plotStr, "gnuplot"))
    {
        plot = Simulator::PLOT_GNUPLOT;
    } else if(boost::iequals(plotStr, "opengl"))
    {
        plot = Simulator::PLOT_OPENGL;
    } else
    {
        plot = Simulator::NO_PLOT;
    }

    Simulator sim = Simulator(
        nX,
        nY,
        nZ,
        timesteps,
        dt,
        state0,
        plot,
        stringToBool(measureStr) ? Simulator::MEASURE : Simulator::NO_MEASURE,
        stringToBool(fftwStr) ? Simulator::FFTW : Simulator::NO_FFTW,
        stringToBool(clfftStr) ? Simulator::CLFFT : Simulator::NO_CLFFT,
        path,
        logger);

    sim.simulate();

    ////measureTimes(logger, state0, path);

    return finish();
}
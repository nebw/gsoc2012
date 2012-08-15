#include "Definitions.h"
#include "CLSimulator.h"
#include "util.h"

#include "cpplog/cpplog.hpp"
#include "gnuplot_i/gnuplot_i.h"

#include <numeric>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>

namespace po = boost::program_options;

void measureTimes(Logger const& logger, state const& state0, const unsigned int timesteps, float dt, boost::filesystem::path const& path)
{
    const unsigned int start = 14;
    const unsigned int powers = 20;

    std::vector<const unsigned int> numNeurons;
    std::vector<const double> avgTimesCalculations;
    std::vector<const double> avgTimesFFTW;
    std::vector<const double> avgTimesClFFT;

    for(int i = start; i < powers; ++i)
    {
        auto neurons = static_cast<const unsigned int>(pow(2.f, i));
        numNeurons.push_back(neurons);

        CLSimulator sim = CLSimulator(
            neurons,
            1,
            1,
            timesteps,
            dt,
            state0,
            CLSimulator::NO_PLOT,
            CLSimulator::MEASURE,
            CLSimulator::FFTW,
            CLSimulator::CLFFT,
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

    std::cout << std::endl << "Results" << std::endl << "=======" << std::endl;
    for (int i = 0; i < powers - start; ++i)
    {
        std::cout << static_cast<const unsigned int>(pow(2.f, (int)(i + start))) << "\t" << avgTimesCalculations[i] << "\t" << avgTimesFFTW[i] << "\t" << avgTimesClFFT[i] << std::endl;
    }

    {
        Gnuplot plot_performance;
        plot_performance.set_title("Performance measurements (NVIDIA NVS 4200M)");
        plot_performance.set_style("linespoints");
        plot_performance.set_xlogscale(2);
        plot_performance.set_xlabel("Neurons");
        plot_performance.set_ylabel("Average execution time (ms)");
        plot_performance.plot_xy(numNeurons, avgTimesCalculations, "Runge-kutta approximations");
        plot_performance.plot_xy(numNeurons, avgTimesFFTW, "Convolution using FFTW");
        plot_performance.plot_xy(numNeurons, avgTimesClFFT, "Convolution using ClFFT");
        getchar();
    }
}

int finish(const int rc)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    _CrtDumpMemoryLeaks();
#endif

    return(rc);
}

int main(int ac, char **av)
{
    float V0, h0, n0, z0, sAMPA0, sNMDA0, xNMDA0, sGABAA0, IApp0, dt;
    unsigned int nX, nY, nZ, timesteps;
    std::string plotStr, measureStr, fftwStr, clfftStr, perfplot;

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
        ("perfplot", po::value<std::string>(&perfplot)->default_value("false"), "measure and plot performance for various network sizes")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(ac, av, desc), vm);
    } catch (po::error&) {
        std::cout << desc << std::endl;
        return finish(1);
    }

    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return finish(1);
    }

    if(stringToBool(clfftStr) && !isPowerOfTwo(nX * nY * nZ))
    {
        std::cout << "Error: numNeurons (nX * nY * nZ) must be radix 2" << std::endl;
        return finish(1);
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

    if (stringToBool(perfplot))
    {
        measureTimes(logger, state0, timesteps, dt, path);
    } else
    {
        CLSimulator::Plot plot;
        if(boost::iequals(plotStr, "gnuplot"))
        {
            plot = CLSimulator::PLOT_GNUPLOT;
        } else if(boost::iequals(plotStr, "opengl"))
        {
            plot = CLSimulator::PLOT_OPENGL;
        } else
        {
            plot = CLSimulator::NO_PLOT;
        }

        CLSimulator sim = CLSimulator(
            nX,
            nY,
            nZ,
            timesteps,
            dt,
            state0,
            plot,
            stringToBool(measureStr) ? CLSimulator::MEASURE : CLSimulator::NO_MEASURE,
            stringToBool(fftwStr) ? CLSimulator::FFTW : CLSimulator::NO_FFTW,
            stringToBool(clfftStr) ? CLSimulator::CLFFT : CLSimulator::NO_CLFFT,
            path,
            logger);

        sim.simulate();
    }

    return finish(0);
}
#include "util.h"
#include "CLWrapper.h"

#include "CL/cl.hpp"

#include "gnuplot_i/gnuplot_i.hpp"

#include "cpplog/cpplog.hpp"

#include "fftw3.h"

#include <iomanip>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
#include <Windows.h>
#endif

template <typename T>
std::vector<T> linSpaceVec(T a, T b, size_t N) {
  T h = (b - a) / static_cast<T>(N-1);
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    *x = val;
  return xs;
}

cpplog::StdErrLogger logger;

namespace cpu
{
    typedef struct
    {
        float V;
        float h;
        float n;
        float z;
        float s_AMPA;
        float x_NMDA;
        float s_NMDA;
        float s_GABAA;
        float I_app;
    } state;

    void initialize() 
    {
        static const unsigned int numNeurons = 100;
        static const unsigned int timesteps = 5000;
        static const float dt = 0.1f;
        
        static const bool plot = true;

        static const float V_0 = -70;
        static const float h_0 = 1.0;
        static const float n_0 = 0;
        static const float z_0 = 0;
        static const float s_AMPA_0 = 0;
        static const float x_NMDA_0 = 0;
        static const float s_NMDA_0 = 0;
        static const float s_GABAA_0 = 0;
        static const float I_app_0 = 1;

        LOG_INFO(logger) << "numNeurons: " << numNeurons;
        LOG_INFO(logger) << "timesteps: " << timesteps;
        LOG_INFO(logger) << "dt: " << dt;

        // old and new state
        state *states = (state *)malloc(2 * numNeurons * sizeof(state));
        float *sumFootprintAMPA = (float *)malloc(numNeurons * sizeof(float));
        float *sumFootprintNMDA = (float *)malloc(numNeurons * sizeof(float));
        float *sumFootprintGABAA = (float *)malloc(numNeurons * sizeof(float));

        // initialize states
        for (unsigned int i = 0; i < numNeurons; ++i)
        {
            states[i].V = V_0;
            states[i].h = h_0;
            states[i].n = n_0;
            states[i].z = z_0;
            states[i].s_AMPA = s_AMPA_0;
            states[i].x_NMDA = x_NMDA_0;
            states[i].s_NMDA = s_NMDA_0;
            states[i].s_GABAA = s_GABAA_0;
            states[i].I_app = I_app_0;

            sumFootprintAMPA[i] = 0;
            sumFootprintNMDA[i] = 0;
            sumFootprintGABAA[i] = 0;
        }
        for (unsigned int i = numNeurons - 1; i < 2 * numNeurons; ++i)
        {
            states[i].V = V_0;
            states[i].h = h_0;
            states[i].n = n_0;
            states[i].z = z_0;
            states[i].s_AMPA = s_AMPA_0;
            states[i].x_NMDA = x_NMDA_0;
            states[i].s_NMDA = s_NMDA_0;
            states[i].s_GABAA = s_GABAA_0;
            states[i].I_app = I_app_0;
        }

        // opencl initialization
        cl_int err;

        CLWrapper cl;

        cl::Buffer states_cl = cl::Buffer(cl.getContext(),
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          2 * numNeurons * sizeof(state),
                                          states,
                                          &err);
        cl::Buffer sumFootprintAMPA_cl(cl.getContext(),
                                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       numNeurons * sizeof(float),
                                       sumFootprintAMPA,
                                       &err);
        cl::Buffer sumFootprintNMDA_cl(cl.getContext(),
                                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       numNeurons * sizeof(float),
                                       sumFootprintNMDA,
                                       &err);
        cl::Buffer sumFootprintGABAA_cl(cl.getContext(),
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        numNeurons * sizeof(float),
                                        sumFootprintGABAA,
                                        &err);

        //err = cl.getQueue().enqueueWriteBuffer(states_cl, CL_TRUE, 0, 2 * numNeurons * sizeof(state), states, NULL, &event);

        //cl.getQueue().finish();

        std::string path(CL_SOURCE_DIR);

        cl::Program program = cl.loadProgram(path + "/kernels.cl");

        

        cl::Kernel kernel_f_dV_dt = cl::Kernel(program, "f_dV_dt", &err);
        err = kernel_f_dV_dt.setArg(0, states_cl);
        err = kernel_f_dV_dt.setArg(1, sumFootprintAMPA_cl);
        err = kernel_f_dV_dt.setArg(2, sumFootprintNMDA_cl);
        err = kernel_f_dV_dt.setArg(3, sumFootprintNMDA_cl);
        err = kernel_f_dV_dt.setArg(4, numNeurons);
        err = kernel_f_dV_dt.setArg(6, dt);

        std::vector<cl::Kernel> kernels;
        cl::Kernel kernel_f_dn_dt = cl::Kernel(program, "f_dn_dt", &err);
        cl::Kernel kernel_f_I_Na_dh_dt = cl::Kernel(program, "f_I_Na_dh_dt", &err);
        cl::Kernel kernel_f_dz_dt = cl::Kernel(program, "f_dz_dt", &err);
        cl::Kernel kernel_f_dsAMPA_dt = cl::Kernel(program, "f_dsAMPA_dt", &err);
        cl::Kernel kernel_f_dxNMDA_dt = cl::Kernel(program, "f_dxNMDA_dt", &err);
        cl::Kernel kernel_f_dsNMDA_dt = cl::Kernel(program, "f_dsNMDA_dt", &err);
        kernels.push_back(kernel_f_dn_dt);
        kernels.push_back(kernel_f_I_Na_dh_dt);
        kernels.push_back(kernel_f_dz_dt);
        kernels.push_back(kernel_f_dsAMPA_dt);
        kernels.push_back(kernel_f_dxNMDA_dt);
        kernels.push_back(kernel_f_dsNMDA_dt);

        BOOST_FOREACH(cl::Kernel kernel, kernels)
        {
            err = kernel.setArg(0, states_cl);
            err = kernel.setArg(1, numNeurons);
            err = kernel.setArg(3, dt);
        }

        cl::Event event; 
        
        //for(unsigned int i = 0; i < numNeurons; ++i)
        //{
        //    std::cout << states[i].V << std::endl;
        //}

            std::vector<float> V_t, h_t, n_t, z_t, sAMPA_t, xNMDA_t, sNMDA_t, I_app_t;
            std::vector<float> spikeTimes, spikeNeuronIndices;
        if(plot)
        {
            V_t.push_back(states[0].V);
            h_t.push_back(states[0].h);
            n_t.push_back(states[0].n);
            z_t.push_back(states[0].z);
            I_app_t.push_back(states[0].I_app);
            sAMPA_t.push_back(states[0].s_AMPA);
            xNMDA_t.push_back(states[0].x_NMDA);
            sNMDA_t.push_back(states[0].s_NMDA);
        }

        std::cout << "Timestep 1/" << timesteps << std::endl;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        timeBeginPeriod(1);
        unsigned long startTime = timeGetTime();
#endif

        for (unsigned int t = 0; t < timesteps - 1; ++t)
        {
            if((t + 2) % (timesteps / 100) == 0)
            {
                std::cout << "Timestep " << t + 2 << "/" << timesteps << std::endl;
            }

            unsigned int ind_old = t % 2;
            unsigned int ind_new = 1 - ind_old;

            err = kernel_f_dV_dt.setArg(5, ind_old);
            BOOST_FOREACH(cl::Kernel kernel, kernels)
            {
                err = kernel.setArg(2, ind_old);
            }

            cl.getQueue().finish();

            err = cl.getQueue().enqueueNDRangeKernel(kernel_f_dV_dt, cl::NullRange, cl::NDRange(numNeurons), cl::NullRange, NULL, &event);
            BOOST_FOREACH(cl::Kernel kernel, kernels)
            {
                err = cl.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numNeurons), cl::NullRange, NULL, &event);
            }

            cl.getQueue().finish();

            err = cl.getQueue().enqueueReadBuffer(states_cl, CL_TRUE, 0, 2 * numNeurons * sizeof(state), states, NULL, &event);

            cl.getQueue().finish();

            /*for(unsigned int i = 0; i < numNeurons; ++i)
            {
                std::cout << "V = " << states[ind_new*numNeurons+i].V << std::endl;
                std::cout << "n = " << states[ind_new*numNeurons+i].n << std::endl;
            }*/

            if(plot)
            {
                V_t.push_back(states[ind_new].V);
                h_t.push_back(states[ind_new].h);
                n_t.push_back(states[ind_new].n);
                z_t.push_back(states[ind_new].z);
                I_app_t.push_back(states[ind_new].I_app);
                sAMPA_t.push_back(states[ind_new].s_AMPA);
                xNMDA_t.push_back(states[ind_new].x_NMDA);
                sNMDA_t.push_back(states[ind_new].s_NMDA);

                for(unsigned int i = 0; i < numNeurons; ++i)
                {
                    if (states[ind_new].V >= 20)
                    {
                        spikeTimes.push_back(t * dt);
                        spikeNeuronIndices.push_back(i);
                    }
                }
            }
        }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        unsigned long elapsedTime = timeGetTime() - startTime;
        LOG_INFO(logger) << "Execution time: " << elapsedTime << "ms" << std::endl;
        timeEndPeriod(1);
#endif

        if(plot)
        {
            Gnuplot plot_V_Iapp_e;
            plot_V_Iapp_e.set_style("lines");
            plot_V_Iapp_e.set_title("Excitatory neuron");
            plot_V_Iapp_e.plot_xy(
                linSpaceVec<float>(0, timesteps * dt, timesteps),
                V_t, "V");
            plot_V_Iapp_e.plot_xy(
                linSpaceVec<float>(0, timesteps * dt, timesteps),
                I_app_t, "I_app");
            Gnuplot plot_hnz_e;
            plot_hnz_e.set_style("lines");
            plot_hnz_e.set_title("Excitatory neuron");
            plot_hnz_e.plot_xy(
                linSpaceVec<float>(0, timesteps * dt, timesteps),
                h_t, "h");
            plot_hnz_e.plot_xy(
                linSpaceVec<float>(0, timesteps * dt, timesteps),
                n_t, "n");
            plot_hnz_e.plot_xy(
                linSpaceVec<float>(0, timesteps * dt, timesteps),
                z_t, "z");
            Gnuplot plot_Syn_e;
            plot_Syn_e.set_style("lines");
            plot_Syn_e.set_title("Excitatory neuron");
            plot_Syn_e.plot_xy(
                linSpaceVec<float>(0, timesteps * dt, timesteps),
                sAMPA_t, "s_AMPA");
            plot_Syn_e.plot_xy(
                linSpaceVec<float>(0, timesteps * dt, timesteps),
                xNMDA_t, "x_NMDA");
            plot_Syn_e.plot_xy(
                linSpaceVec<float>(0, timesteps * dt, timesteps),
                sNMDA_t, "s_NMDA");
            Gnuplot plot_Spikes;
            plot_Spikes.set_title("Spikes");
            plot_Spikes.set_style("points");
            plot_Spikes.set_xrange(0, timesteps * dt);
            plot_Spikes.set_yrange(0, numNeurons - 1);
            plot_Spikes.plot_xy(
                spikeTimes, spikeNeuronIndices, "Excitatory Spikes");
            getchar();
        } else {
            getchar();
        }
    }
}

typedef struct
{
    double *real;
    double *imag;
} clFFT_SplitComplexDouble;


int main(int argc, char **argv)
{
    cpu::initialize();

    exit(0);
}

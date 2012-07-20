#include "Definitions.h"
#include "Simulator.h"

#include "cpplog/cpplog.hpp"

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

int main(int argc, char **argv)
{
    auto logger = boost::make_shared<cpplog::StdErrLogger>();
    state state0;
    state0.V = -70;
    state0.h = 1.0;
    state0.n = 0.0;
    state0.z = 0.0;
    state0.s_AMPA = 0.0;
    state0.s_NMDA = 0.0;
    state0.x_NMDA = 0.0;
    state0.s_GABAA = 0.0;
    state0.I_app = 1.0;

    auto path = boost::filesystem3::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    Simulator sim = Simulator(
        800,
        5000,
        0.1f,
        state0,
        false,
        true,
        path,
        logger);

    sim.simulate();

    exit(0);
}
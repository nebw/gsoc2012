#include "stdafx.h"

#include "CPUSimulator.h"

void CPUSimulator::step()
{
    throw std::exception("The method or operation is not implemented.");
}

void CPUSimulator::simulate()
{
    throw std::exception("The method or operation is not implemented.");
}

std::unique_ptr<state[]> const& CPUSimulator::getCurrentStates() const
{
    throw std::exception("The method or operation is not implemented.");
}

std::unique_ptr<float[]> const& CPUSimulator::getCurrentSumFootprintAMPA() const
{
    throw std::exception("The method or operation is not implemented.");
}

std::unique_ptr<float[]> const& CPUSimulator::getCurrentSumFootprintNMDA() const
{
    throw std::exception("The method or operation is not implemented.");
}

std::unique_ptr<float[]> const& CPUSimulator::getCurrentSumFootprintGABAA() const
{
    throw std::exception("The method or operation is not implemented.");
}

std::vector<unsigned long> CPUSimulator::getTimesCalculations() const
{
    throw std::exception("The method or operation is not implemented.");
}

std::vector<unsigned long> CPUSimulator::getTimesFFTW() const
{
    throw std::exception("The method or operation is not implemented.");
}

std::vector<unsigned long> CPUSimulator::getTimesClFFT() const
{
    throw std::exception("The method or operation is not implemented.");
}

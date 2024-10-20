//
// Created by bryce on 12/10/24.
//

#ifndef PROGRAM_H
#define PROGRAM_H
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "Parameters.h"

class Program {
private:
    std::array<double, Parameters::NUM_REGISTERS> registers;
    int getRandomNumber(int min, int max);

public:
    std::vector<uint16_t> instructions;

    void addRandomInstruction();

    Program();

    void execute(const std::vector<double> &features);

    double predict(const std::vector<double> &features);

    void displayInstructions() const;

    void displayRegisters() const;

    void displayCode() const;

    std::vector<uint16_t> __getstate__() const;

    void __setstate__(std::vector<uint16_t> state);
};

#endif //PROGRAM_H

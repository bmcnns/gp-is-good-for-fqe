//
// Created by bryce on 12/10/24.
//

#ifndef PARAMETERS_H
#define PARAMETERS_H

class Parameters {
public:
    static const int NUM_REGISTERS = 8;
    static const int NUM_FEATURES = 16;
    static const int MAX_PROGRAM_LENGTH = 100;
    static const int NUM_OP_CODES = 6;
    inline static const double DELETE_INSTRUCTION_PROBABILITY = 0.7;
    inline static const double ADD_INSTRUCTION_PROBABILITY = 0.7;
    inline static const double MUTATE_INSTRUCTION_PROBABILITY = 0.65;
    inline static const double SWAP_INSTRUCTION_PROBABILITY = 1.0;
};

#endif //PARAMETERS_H
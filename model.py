import random
from copy import deepcopy
from uuid import uuid4

import numpy as np
from tqdm import tqdm

import parameters
from linear_gp import Program, Mutator

class Model:
    def __init__(self):
        self.population = []
        self.fitnesses = {}

        for _ in range(parameters.POPULATION_SIZE):
            self.population.append((Program(), uuid4()))

    def fit(self, four_tuples, generation):
        for individual, individual_id in tqdm(self.population):
            inputs = []
            targets = []

            for observation, action, reward, next_observation in four_tuples:
                state_action = np.concatenate([[action], observation.flatten()])
                inputs.append(state_action)

                next_state_actions = [np.concatenate([[a], next_observation.flatten()]) for a in range(3)]

                next_q_values = list(map(individual.predict, next_state_actions))

                target = reward + parameters.DISCOUNT_FACTOR * np.max(next_q_values)
                targets.append(target)

            predictions = [individual.predict(input) for input in inputs]

            predictions = np.array(predictions)
            targets = np.array(targets)

            mse = np.mean((targets - predictions) ** 2)

            self.fitnesses[individual_id] = -mse

        print(f"Generation {generation}, Average MSE: {np.median(np.array(list(self.fitnesses.values())))}")

        # we sort individuals by their fitnesses x[1] is the key of the individual
        self.population.sort(key=lambda x: self.fitnesses[x[1]], reverse=True)

        self.population = self.population[:int(parameters.POP_GAP * parameters.POPULATION_SIZE)]

        parents = random.choices(self.population, k=int(parameters.POP_GAP * parameters.POPULATION_SIZE))
        for parent, parent_id in parents:
            child = deepcopy(parent)

            Mutator.mutateProgram(child)
            self.population.append((child, uuid4()))
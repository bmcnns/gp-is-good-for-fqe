import random
import numpy as np
import gymnasium as gym
import minigrid
import pickle

env = gym.make("MiniGrid-FourRooms-v0", render_mode="human")
observation, info = env.reset()

with open('results/four-rooms-gen-197.pkl', 'rb') as f:
    model = pickle.load(f)

for _ in range(10000):
    next_state_actions = [np.concatenate([[a], observation['image'].flatten()]) for a in range(3)]

    best_individual, best_individual_id = model.population[0]
    print (f"best individual has fitness: {model.fitnesses[best_individual_id]}")

    next_q_values = list(map(best_individual.predict, next_state_actions))
    print(next_q_values)

    best_action = next_state_actions[np.argmax(next_q_values)][0]
    print(best_action)

    observation, reward, terminated, truncated, info = env.step(best_action)

    if terminated or truncated:
       observation, info = env.reset()

env.close()

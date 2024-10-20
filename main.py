import minari
from tqdm import tqdm

from model import Model
import pickle
import os


def get_batch_data(dataset):
    four_tuples = []
    for episode in dataset.iterate_episodes():
        observations = episode.observations['image']
        num_steps = len(observations) - 1

        for i in range(num_steps):
            observation = observations[i]
            action = episode.actions[i]
            reward = episode.rewards[i]
            next_observation = observations[i + 1]

            four_tuples.append((observation, action, reward, next_observation))

    return four_tuples

dataset = minari.load_dataset("D4RL/minigrid/fourrooms-v0")
data = get_batch_data(dataset)

model = Model()

directory = 'results'
os.makedirs(directory, exist_ok=True)

for i in tqdm(range(200)):
    model.fit(data, generation=i)

    with open(f'results/four-rooms-gen-{i}.pkl', 'wb') as f:
        pickle.dump(model, f)

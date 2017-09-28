import random
import numpy as np
from mss import mss
from PIL import Image
from tile_encoder import TileEncoder
from q_score import QScore
from thought_lstm import ThoughtLSTM



sct = mss()

# Simply grab a 64x64 pixel array from the screen 
# with top left corner at (x,y).
def single_tile(x, y):
    sct_img = sct.grab({'top': y, 'left': x, 'width': 32, 'height': 32})
    return np.array(sct_img)[:,:,:3]



# Agent Parameters.
x = 0
y = 0
step = 4
long_mem = 2500
batch_size = 64

# Tile encoder parameters.
encoding_size = 16

# Thought LSTM parameters.
short_mem = 16

# Q-score parameters.
num_actions = 4
action_vectors = [[0]*num_actions for i in range(num_actions)]
for i in range(num_actions): action_vectors[i][i] = 1



# The networks.
tile_encoder = TileEncoder(encoding_size)
thought_lstm = ThoughtLSTM(encoding_size + num_actions, encoding_size, short_mem)
q_score = QScore()

# Memory objects.
tile_memory = list()
thought_memory = list()
action_memory = list()
brain_memory = list() # Brain memory is merely the thought memories and action memories merged.

while True:
    for _ in range(batch_size):
        # Observe.
        tile = single_tile(x, y)
        tile_memory.append(tile)

        # Think.
        thought = tile_encoder.encode([tile])[0]
        thought_memory.append(thought)

        # Act.
        #expected_scores = q_score.predict(#TODO))
        expected_scores = [random.random() for _ in range(num_actions)]
        action_idx = np.argmax(expected_scores)
        action_vec = action_vectors[action_idx]
        action_memory.append(action_vec)

        # Memory.
        brain_memory.append(np.concatenate((thought, action_vec)))

        # Move.
        x += [0,1,0,-1][action_idx] * step
        y += [-1,0,1,0][action_idx] * step

        # Enforce boundries.
        x = max(0, min(1280 - 32, x))
        y = max(0, min(800 - 32, y))

    # Forget.
    tile_memory = tile_memory[-long_mem:]
    thought_memory = thought_memory[-long_mem:]

    # Create the necesary vectors for the LSTM and Q-score
    brain_sequences = [brain_memory[i:i+short_mem] for i in range(len(brain_memory) - short_mem)]
    resulting_states = thought_memory[short_mem:]

    # Calculate the score for the last batch.
    expected_thoughts = thought_lstm.predict(brain_sequences)
    score_memory = [np.mean((t-e)**2) for t, e in zip(thought_memory[::-1], expected_thoughts[::-1])][::-1]

    # Train the thought LSTM.
    thought_lstm.fit(brain_sequences, resulting_states)
    thought_lstm.save()

    #TODO
    # Train the Q-score.

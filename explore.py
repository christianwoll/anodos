import random
import numpy as np
from mss import mss
from PIL import Image
import matplotlib.pyplot as plt
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
x = 640 - 16
y = 400 - 16
step = 4
long_mem = 2000
batch_size = 64

# Tile encoder parameters.
encoding_size = 16

# Thought LSTM parameters.
short_mem = 16

# Q-score parameters.
num_actions = 4
action_vectors = [[0]*num_actions for i in range(num_actions)]
for i in range(num_actions): action_vectors[i][i] = 1
lam = 0.5



# The networks.
tile_encoder = TileEncoder(encoding_size)
thought_lstm = ThoughtLSTM(short_mem, encoding_size + num_actions, encoding_size)
q_score = QScore(short_mem, encoding_size + num_actions)

# Memory objects.
tile_memory = list()
thought_memory = list()
action_memory = list()
brain_memory = list() # Brain memory is merely the thought memories and action memories merged.
expected_score_memory = list()

batch_num = 0

while True:
    batch_num += 1
    print(' ')
    print('Batch: ' + str(batch_num))

    for _ in range(batch_size):
        # Observe.
        tile = single_tile(x, y)
        tile_memory.append(tile)

        # Think.
        thought = tile_encoder.encode([tile])[0]
        thought_memory.append(thought)

        # Act.
        if len(brain_memory) <= short_mem - 1:
            # Random actions at the start
            action_idx = random.randrange(num_actions)
        else:
            # Get the expected value of the scores corresponding to each action.
            possible_sequences = [brain_memory[-short_mem+1:] + [np.concatenate((thought, v))] for v in action_vectors]
            expected_scores = q_score.predict(possible_sequences)
            expected_scores = [random.random() for _ in range(num_actions)]

            # Do the action with the highest score.
            action_idx = np.argmax(expected_scores)

            # Store the highest score training the Q-score.
            expected_score = expected_scores[action_idx]
            expected_score_memory.append(expected_score)


        # Act.
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
    action_memory = action_memory[-long_mem:]
    brain_memory = brain_memory[-long_mem:]

    expected_score_memory = expected_score_memory[-long_mem+short_mem:]

    # Create the necesary vectors for the LSTM and Q-score
    brain_sequences = [brain_memory[i:i+short_mem] for i in range(len(brain_memory) - short_mem)]
    resulting_states = thought_memory[short_mem:]

    # Calculate the reward.
    expected_thoughts = thought_lstm.predict(brain_sequences)
    rewards = [np.mean((t-e)**2) for t, e in zip(thought_memory[::-1], expected_thoughts[::-1])][::-1]

    print('Training the thought LSTM...')

    # Train the thought LSTM.
    thought_lstm.fit(brain_sequences, resulting_states)
    thought_lstm.save()

    print('Training the Q-score...')

    # Train the Q-score.
    target_scores = [lam * r + (1 - lam) * score for r, score in zip(rewards, expected_score_memory)]
    q_score.fit(brain_sequences, target_scores)
    q_score.save()

    if batch_num % 5 == 0:
        print('Training tile encoder...')

        tile_encoder.fit(tile_memory)
        tile_encoder.save()

    """
    if batch_num % 10 == 0:
        im = plt.imshow(tile_memory[-10 * batch_size])
        for tile in tile_memory[-10 * batch_size:]:
            im.set_data(tile)
            plt.pause(0.00001)
        plt.show()
    """

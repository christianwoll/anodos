import random
import numpy as np
from mss import mss
from PIL import Image
from encoder import TopEncoder, MidEncoder, BumEncoder

sct = mss()

# BUG:
# The dict is modified in place by mss so make a copy for each call.
mon = {'top': 0, 'left': 0, 'width': 1280, 'height': 800}

def grab_tiles(num_tiles, tile_size=(64,64)):
    sct_img = sct.grab(dict(mon))

    print('<screenshot taken>')

    pixels = np.array(sct_img)

    tiles = []
    for _ in range(num_tiles):
        i = random.randrange(pixels.shape[0] - tile_size[0])
        j = random.randrange(pixels.shape[1] - tile_size[1])
        tile = pixels[i:i+tile_size[0],j:j+tile_size[1]]
        Image.fromarray(tile).show()
        tiles.append(tile)

    return tiles

top_encoder = TopEncoder()
mid_encoder = MidEncoder()
bum_encoder = BumEncoder()

epoch = 1;tiles = []
while True:
    print('Beginning epoch ' + str(epoch))

    tiles = [img for img in tiles if random.random() > 1.0 / 100]
    tiles += grab_tiles(100)

    print(str(len(tiles)) + ' tiles in corpus.')

    print('Training top tile encoder...')
    top_encoder.fit(tiles)
    top_ncoder.save()

    print('Training mid tile encoder...')
    chips = top_encoder.encode(tiles)
    mid_encoder.fit(chips)
    mid_encoder.save()

    print('Training bum tile encoder...')
    chips = mid_encoder.encode(chips)
    bum_encoder.fit(chips)
    bum_encoder.save()

    epoch += 1

    print(' ')

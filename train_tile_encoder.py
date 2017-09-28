import random
import numpy as np
from mss import mss
from PIL import Image
from tile_encoder import TileEncoder

sct = mss()

def grab_tiles(num_tiles, tile_size=(64,64)):
    sct_img = sct.grab({'top': 0, 'left': 0, 'width': 1280, 'height': 800})

    print('<screenshot taken>')

    pixels = np.array(sct_img)[:,:,:3]

    tiles = []
    for _ in range(num_tiles):
        i = random.randrange(pixels.shape[0] - tile_size[0])
        j = random.randrange(pixels.shape[1] - tile_size[1])
        tile = pixels[i:i+tile_size[0],j:j+tile_size[1]]
        tiles.append(tile)

    return tiles



tile_encoder = TileEncoder()

epoch = 1;tiles = []
while True:
    print('Beginning epoch ' + str(epoch))

    tiles = [img for img in tiles if random.random() > 1.0 / 100]
    tiles += grab_tiles(100)

    print(str(len(tiles)) + ' tiles in corpus.')

    print('Training tile encoder...')
    tile_encoder.fit(tiles)
    tile_encoder.save()

    epoch += 1

    print(' ')

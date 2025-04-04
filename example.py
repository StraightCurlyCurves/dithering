import cv2 as cv
import numpy as np
import time

from dithering import Dither


if __name__ == '__main__':
    image = cv.imread('test_img.png')
    assert image is not None, "Image not found or unable to read."

    new_filename = 'index_image.dim.png' # .png ending helps the OS to see the png file at the beginning of the file

    # create a dithered image
    dither = Dither()
    print('Dithering...')
    tick = time.perf_counter()
    dither.dither(image)
    tock = time.perf_counter()
    print(f'Dithering time: {tock - tick:.3f} s')

    # Convert to index image
    tick = time.perf_counter()
    dither.to_index()
    tock = time.perf_counter()
    print(f'To index time: {tock - tick:.3f} s')

    # save index image to file
    thumbnail = cv.resize(image, (image.shape[1] // 10, image.shape[0] // 10))
    tick = time.perf_counter()
    dither.save_index_image(new_filename, thumbnail=thumbnail)
    tock = time.perf_counter()
    print(f'Save index image time: {tock - tick:.3f} s')

    # load index image from file
    tick = time.perf_counter()
    dither.load_index_image(new_filename)
    tock = time.perf_counter()
    print(f'Load index image time: {tock - tick:.3f} s')
    
    # convert to dithered image
    tick = time.perf_counter()
    dither.from_index()
    tock = time.perf_counter()
    print(f'From index time: {tock - tick:.3f} s')

    # show dithered image
    cv.imshow('Dithered Image', dither.get_dithered_image())
    key = cv.waitKey(0)
    cv.destroyAllWindows()
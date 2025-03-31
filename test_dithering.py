import os

from dithering import Dither
import numpy as np
import cv2 as cv

def test_find_nearest_color():
    palette = np.array([[0, 0, 0], [255, 255, 255]])
    dither = Dither(palette)
    color = np.array([128, 128, 128])
    assert np.all(dither._find_nearest_color(color) == [255, 255, 255])

    dither = Dither(palette)
    color = np.array([127, 127, 127])
    assert np.all(dither._find_nearest_color(color) == [0, 0, 0])

    dither = Dither(palette)
    color = np.array([129, 129, 129])
    assert np.all(dither._find_nearest_color(color) == [255, 255, 255])

def test_save_index_image():
    image = cv.imread('test_img.png')
    assert image is not None, "Image not found or unable to read."
    dither_save = Dither()
    dither_save.dither(image)
    dither_save.to_index()
    dither_save.save_index_image('test.dim')
    assert os.path.exists('test.dim')
    dither_load = Dither()
    dither_load.load_index_image('test.dim')
    dither_load.from_index()
    index_is_same = np.all(dither_load._index_image == dither_save._index_image)
    dithered_is_same = np.all(dither_load._dithered_image == dither_save._dithered_image)
    os.remove('test.dim')
    assert index_is_same, "Index image mismatch."
    assert dithered_is_same, "Dithered image mismatch."

def test_save_index_image_with_thumbnail():
    image = cv.imread('test_img.png')
    assert image is not None, "Image not found or unable to read."
    thumbnail = cv.resize(image, (image.shape[1] // 5, image.shape[0] // 5))
    dither_save = Dither()
    dither_save.dither(image)
    dither_save.to_index()
    dither_save.save_index_image('test.dim', thumbnail=thumbnail)
    assert os.path.exists('test.dim')
    thumbnail_loaded = cv.imread('test.dim', cv.IMREAD_UNCHANGED)
    dither_load = Dither()
    dither_load.load_index_image('test.dim')
    dither_load.from_index()
    index_is_same = np.all(dither_load._index_image == dither_save._index_image)
    dithered_is_same = np.all(dither_load._dithered_image == dither_save._dithered_image)
    thumbnail_is_same = np.all(thumbnail_loaded == thumbnail)
    os.remove('test.dim')
    assert thumbnail_is_same, "Thumbnail mismatch."
    assert index_is_same, "Index image mismatch."
    assert dithered_is_same, "Dithered image mismatch."

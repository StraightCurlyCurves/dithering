import time
import os
import zlib

import cv2 as cv
import numpy as np
import numba as nb

class Dither:

    def __init__(self, palette = None):
        '''
        Initialize Dither object.

        Keyword arguments:
        image (np.ndarray): Image to dither. Default is None.
        palette (np.ndarray): Color palette. Default is None.
        dithered_image (np.ndarray): Dithered image. Default is None.
        index_image (np.ndarray): Index image. Default is None.
        '''
        self._palette: np.ndarray = None
        self._dithered_image: np.ndarray = None
        self._index_image: np.ndarray = None

        if palette is not None:
            self.set_palette(palette)
        else:
            palette = np.array([
                [0, 0, 0],
                [255, 255, 255],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],
                [255, 0, 255],
                ]) # KWRGBYCM
            self.set_palette(palette)

    def set_palette(self, palette: np.ndarray) -> None:
        self._palette = np.copy(palette)
        self._dithered_image = None
        self._index_image = None

    def get_palette(self) -> np.ndarray:
        return np.copy(self._palette)

    def get_dithered_image(self) -> np.ndarray:
        if self._dithered_image is None:
            raise ValueError('Dithered image is not set. Use calculate_floyd_steinberg() method to calculate dithered image.')
        return np.copy(self._dithered_image)
    
    def get_index_image(self) -> np.ndarray:
        if self._index_image is None:
            raise ValueError('Index image is not set. Use calculate_index_image() method to calculate index image from dithered image.')
        return np.copy(self._index_image)

    def dither(self, image: np.ndarray) -> None:
        '''
        Calculate dithered image from image.
        '''
        self._dithered_image = self._get_floyd_steinberg(image, self._palette)
        self._index_image = None

    def to_index(self) -> None:
        '''
        Calculate index image from dithered image.
        '''
        if self._dithered_image is None:
            raise ValueError('Dithered image is not set')
        self._index_image = np.zeros(self._dithered_image.shape[:2], dtype=np.uint8)
        palette_size = self._palette.shape[0]
        unique_colors = np.unique(self._dithered_image.reshape(-1, 3), axis=0).shape[0]
        if unique_colors > palette_size:
            raise ValueError('Palette must have at least as many colors as the dithered image')
        for i, color in enumerate(self._palette):
            mask = np.all(self._dithered_image == color, axis=2)
            self._index_image[mask] = i

    def from_index(self) -> None:
        '''
        Calculate dithered image from index image.
        '''
        if self._index_image is None:
            raise ValueError('Index image is not set')
        self._dithered_image = self._palette[self._index_image]

    def save_index_image(self, filename: str, thumbnail: np.ndarray = None) -> None:
        '''
        Save index image to file.

        Parameters:
        filename (str): Name of the file to save the index image to.
        thumbnail (np.ndarray): Thumbnail image will be saved as png at the beginning of the file.
        
        Fileformat:
        See fileformat definition in README.md for details.
        '''
        if self._index_image is None:
            raise ValueError('Index image is not set')
        if os.path.exists(filename):
            os.remove(filename)
        if thumbnail is not None:
            _, buffer = cv.imencode('.png', thumbnail)
            with open(filename, 'wb') as file:
                file.write(buffer)
        with open(filename, 'a+b') as file:
            ### Start marker
            marker = 0xC55D
            file.write(marker.to_bytes(2, byteorder='big'))
            data_start = file.tell()

            ### Format version number
            version = 1
            file.write(version.to_bytes(1, byteorder='big'))

            ### Header
            header = 'Dithered index image'.encode('utf-8')
            header_size = len(header)
            file.write(header_size.to_bytes(2, byteorder='big'))
            file.write(header)

            ### Dimensions
            height, width = self._index_image.shape
            file.write(height.to_bytes(2, byteorder='big'))
            file.write(width.to_bytes(2, byteorder='big'))

            ### Number of colors in color palette
            number_of_colors = self._palette.shape[0]
            file.write(number_of_colors.to_bytes(1, byteorder='big'))

            ### Color palette
            for color in self._palette:
                for channel in color:
                    channel = int(channel)
                    file.write(channel.to_bytes(1, byteorder='big'))

            ### Color palette size in bits
            color_palette_size_in_bits = int(np.floor(np.log2(number_of_colors)) + 1*(np.log2(number_of_colors) > np.floor(np.log2(number_of_colors))))
            assert color_palette_size_in_bits <= 8, 'Color palette size must be less than or equal to 8 bits'
            file.write(color_palette_size_in_bits.to_bytes(1, byteorder='big'))
            
            ### Image data
            assert self._index_image.dtype == np.uint8
            bit_arr = np.unpackbits(self._index_image).reshape(-1, 8)[..., -color_palette_size_in_bits:]
            byte_arr = np.packbits(bit_arr)
            file.write(byte_arr)

            ### Calculate CRC checksum
            file.seek(data_start)
            data = file.read()
            checksum = zlib.crc32(data)
            file.write(checksum.to_bytes(4, byteorder='big'))

            ### End marker
            file.write(marker.to_bytes(2, byteorder='big'))

    def load_index_image(self, filename: str) -> None:
        '''
        Load index image from file.

        Parameters:
        filename (str): Name of the file to load the index image from.
        '''
        with open(filename, 'rb') as file:
            ### If file is a PNG image, move cursor to the end of the png data
            png_signature = b"\x89PNG\r\n\x1a\n"
            if file.read(8) == png_signature:
                while True:
                    # Read chunk length (4 bytes, big-endian)
                    chunk_length_data = file.read(4)
                    if len(chunk_length_data) < 4:
                        raise ValueError("Unexpected end of file")

                    chunk_length = int.from_bytes(chunk_length_data, "big")

                    # Read chunk type (4 bytes)
                    chunk_type = file.read(4)
                    if len(chunk_type) < 4:
                        raise ValueError("Unexpected end of file")

                    # Check if it's the IEND chunk
                    if chunk_type == b"IEND":
                        # Move the cursor to the end of the IEND chunk (data + CRC)
                        file.seek(4, os.SEEK_CUR)  # Skip CRC (4 bytes)
                        break

                    # Skip the chunk data and CRC (length + 4 bytes CRC)
                    file.seek(chunk_length + 4, os.SEEK_CUR)
            else:
                file.seek(0)

            ### Start marker
            marker = int.from_bytes(file.read(2), byteorder='big')
            assert marker == 0xC55D, 'Start marker is incorrect'

            ### Check CRC checksum
            data_start = file.tell()
            file.seek(-6, os.SEEK_END)  # Stop before the checksum (4 bytes) and end marker (2 bytes)
            data_end = file.tell()
            file.seek(data_start)
            data = file.read(data_end - data_start)
            checksum = int.from_bytes(file.read(4), byteorder='big')
            assert zlib.crc32(data) == checksum, 'CRC checksum mismatch'
            file.seek(data_start)

            ### Format version number
            version = int.from_bytes(file.read(1), byteorder='big')
            assert version == 1, 'Format version is incorrect'

            ### Header
            header_size = int.from_bytes(file.read(2), byteorder='big')
            header = file.read(header_size).decode('utf-8')
            print(f'Header: {header}')
            # file.seek(header_size, os.SEEK_CUR)

            ### Dimensions
            height = int.from_bytes(file.read(2), byteorder='big')
            width = int.from_bytes(file.read(2), byteorder='big')

            ### Number of colors in color palette
            number_of_colors = int.from_bytes(file.read(1), byteorder='big')

            ### Color palette
            color_palette = np.zeros((number_of_colors, 3), dtype=np.uint8)
            for i in range(number_of_colors):
                for j in range(3):
                    color_palette[i, j] = int.from_bytes(file.read(1), byteorder='big')

            ### Color palette size in bits
            color_palette_size_in_bits = int.from_bytes(file.read(1), byteorder='big')

            ### Image data
            buffer = file.read()
            byte_array = np.frombuffer(buffer, dtype=np.uint8)[:-2]
            bit_array = np.unpackbits(byte_array)[:height*width*color_palette_size_in_bits].reshape(height, width, color_palette_size_in_bits)
            indices = np.zeros((height, width, 8), dtype=np.uint8)
            indices[..., -color_palette_size_in_bits:] = bit_array
            indices = np.packbits(indices).reshape(height, width)

            ### End marker (from buffer)
            marker = int.from_bytes(buffer[-2:], byteorder='big')
            assert marker == 0xC55D, 'End marker is incorrect'

            ### Set palette and index image
            self._palette = color_palette
            self._index_image = indices

    def _find_nearest_color(self, color):
        vectors = self._palette.astype(np.float32) - color.astype(np.float32)
        distances = np.sqrt(np.sum((vectors) ** 2, axis=1))
        return self._palette[np.argmin(distances)]
    
    @staticmethod
    @nb.njit(fastmath=True)
    def _get_floyd_steinberg(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        if len(image.shape) >= 2:
            height = image.shape[0]
            width = image.shape[1]
        else:
            raise ValueError('Image must be either 2D or 3D')
        dithered_image = np.copy(image).astype(np.float64)
        for y in range(height):
            for x in range(width):
                    old_pixel = np.copy(dithered_image[y, x])
                    diff = palette - old_pixel
                    distances = np.sqrt(np.sum((diff) ** 2, axis=1))
                    new_pixel = palette[np.argmin(distances)]
                    dithered_image[y, x] = new_pixel
                    quant_error = old_pixel - new_pixel

                    if x + 1 < width:
                        dithered_image[y, x + 1] += quant_error * 7 / 16
                    if y + 1 < height:
                        if x > 0:
                            dithered_image[y + 1, x - 1] += quant_error * 3 / 16
                        dithered_image[y + 1, x] += quant_error * 5 / 16
                        if x + 1 < width:
                            dithered_image[y + 1, x + 1] += quant_error * 1 / 16
        return dithered_image.astype(np.uint8)

if __name__ == '__main__':
    image = cv.imread('test_img.png')
    assert image is not None, "Image not found or unable to read."

    new_filename = 'index_image.dim.png' # .png ending helps the OS to see the png file at the beginning of the file

    dither = Dither()
    print('Dithering...')
    tick = time.perf_counter()
    dither.dither(image)
    tock = time.perf_counter()
    print(f'Dithering time: {tock - tick:.3f} s')
    cv.imshow('Dithered Image', dither.get_dithered_image())
    key = cv.waitKey(0)
    cv.destroyAllWindows()

    tick = time.perf_counter()
    dither.to_index()
    tock = time.perf_counter()
    print(f'To index time: {tock - tick:.3f} s')

    thumbnail = cv.resize(image, (image.shape[1] // 10, image.shape[0] // 10))
    tick = time.perf_counter()
    dither.save_index_image(new_filename, thumbnail=thumbnail)
    tock = time.perf_counter()
    print(f'Save index image time: {tock - tick:.3f} s')

    tick = time.perf_counter()
    dither.load_index_image(new_filename)
    tock = time.perf_counter()
    print(f'Load index image time: {tock - tick:.3f} s')
    
    tick = time.perf_counter()
    dither.from_index()
    tock = time.perf_counter()
    print(f'From index time: {tock - tick:.3f} s')
    cv.imshow('Dithered Image', dither.get_dithered_image())
    key = cv.waitKey(0)
    cv.destroyAllWindows()
# Dithering

This project implements Floyd-Steinberg dithering and an efficient indexed image format that reduces storage by using the minimum number of bits per pixel, based on the palette size.

# Custom Dithered Image File Format

**Big-Endian Clarification:** All multi-byte fields (start marker, end marker, header size, height, width, and checksum) are stored in **big-endian** format. This ensures consistency across all structured parts of the file.

| **Offset**            | **Size**                           | **Description**                                                                 |
|-----------------------|------------------------------------|---------------------------------------------------------------------------------|
| -                     | -                                  | **PNG Image (Optional)**: PNG file at the beginning of the file. Can be used as a preview of the image for applications that do not support the custom format. |
| 0                     | 2 bytes                            | **Start Marker**: Magic number `0xC55D` (used to identify the file format).     |
| 2                     | 1 byte                             | **Version**: Format version number.                                             |
| 3                     | 2 bytes                            | **Header Size**: Length of the header in bytes (excluding the start marker).    |
| 5                     | *Header Size* bytes                | **Header**: UTF-8 encoded text description (e.g., author, title, etc.).         |
| 5 + Header Size       | 2 bytes                            | **Height**: Number of rows (image height) in pixels (**big-endian**).           |
| 7 + Header Size       | 2 bytes                            | **Width**: Number of columns (image width) in pixels (**big-endian**).          |
| 9 + Header Size       | 1 byte                             | **Palette Count**: Number of colors in the color palette (1 - 256).             |
| 10 + Header Size      | `3 × Palette Count` bytes          | **Palette Data**: RGB values of the color palette (3 bytes per color: R, G, B). |
| 10 + Header Size + (3 × Palette Count) | 1 byte            | **Bits per Pixel**: Number of bits used per pixel index (1 - 8 bits).           |
| 11 + Header Size + (3 × Palette Count) | `ceil(Height × Width × Bits per Pixel ÷ 8)` bytes | **Image Data**: Pixel indices, bit-packed left-padded.                           |
| End of Image Data   | 4 bytes                            | **Checksum (CRC32)**: Integrity check of all previous bytes, starting at offset 2 (Version).    |
| End of Image Data + 4 | 2 bytes                           | **End Marker**: Terminator `0xC55D` (ensures the file ends properly).            |

---
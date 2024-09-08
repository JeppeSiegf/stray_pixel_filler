import cv2
import numpy as np
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the cyan color (BGRA format for OpenCV)
cyan_color = np.array([0, 0, 0, 0], dtype=np.uint8)


# Efficient function to check unique pixel status using Numba
@jit(nopython=True, parallel=True)
def check_unique_pixel(image, y, x):
    if x < 1 or x >= image.shape[1] - 1 or y < 1 or y >= image.shape[0] - 1:
        return False

    center_pixel = image[y, x]
    neighbors = [
        image[y - 1, x - 1], image[y - 1, x], image[y - 1, x + 1],
        image[y, x - 1], image[y, x + 1],
        image[y + 1, x - 1], image[y + 1, x], image[y + 1, x + 1]
    ]

    for neighbor in neighbors:
        if not np.array_equal(center_pixel, neighbor):
            return True

    return False


# Process a section of the image
def process_section(image, y_start, y_end, x_start, x_end):
    section = image[y_start:y_end, x_start:x_end]

    # Create a mask for unique pixels
    unique_mask = np.zeros(section.shape[:2], dtype=bool)

    for y in prange(section.shape[0]):
        for x in prange(section.shape[1]):
            if check_unique_pixel(section, y, x):
                unique_mask[y, x] = True

    # Apply the mask to set unique pixels to cyan
    section[unique_mask] = cyan_color

    return section, y_start, x_start


# Load the image
input_image_path = "C:/Users/Jeppe/Pictures/Uplay/aoe_png.png"
image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

height, width = image.shape[:2]

# Define tile size
tile_size = (1000, 1000)  # Adjust based on available memory

tiles = []
for y in range(0, height, tile_size[1]):
    for x in range(0, width, tile_size[0]):
        y_end = min(y + tile_size[1], height)
        x_end = min(x + tile_size[0], width)
        tiles.append((y, y_end, x, x_end))

total_tiles = len(tiles)
print(f"Total number of tiles to process: {total_tiles}")

# Process tiles in parallel
results = []
with ThreadPoolExecutor() as executor:
    future_to_tile = {executor.submit(process_section, image, *tile): tile for tile in tiles}

    # Track the number of completed tiles
    for i, future in enumerate(as_completed(future_to_tile)):
        tile_info = future_to_tile[future]
        try:
            processed_section, y_start, x_start = future.result()
            results.append((processed_section, y_start, x_start))
            print(f"Tile processed: ({x_start}, {y_start}) [{i + 1}/{total_tiles}]")
        except Exception as exc:
            print(f"Tile ({tile_info[2]}, {tile_info[0]}) generated an exception: {exc}")

# Reconstruct the full image from tiles
full_image = np.zeros_like(image)
for processed_section, y_start, x_start in results:
    full_image[y_start:y_start + tile_size[1], x_start:x_start + tile_size[0]] = processed_section

# Save the modified image
output_image_path = "output_image.png"
cv2.imwrite(output_image_path, full_image)

print(f"Processing complete. Saved the modified file as {output_image_path}")

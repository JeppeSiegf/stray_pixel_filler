import cv2
import numpy as np
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor, as_completed

# Efficient function to check unique pixel status using Numba
# Define the black color (BGRA format for OpenCV)

input_image_path = "input_image.png"
output_image_path = "output_image.png"

# Define the target color for stray pixels (BGRA format for OpenCV)
target_color = np.array([0, 0, 0, 255], dtype=np.uint8) # Default black

# Define tile size for batching
tile_size = (1000, 1000)  # Adjust based on available memory


# Function to check if the pixel should be black
@jit(nopython=True, parallel=True)
def check_neighbors(image, y, x):
    height, width = image.shape[:2]

    # Check if the pixel is at the border
    if x < 1 or x >= width - 1 or y < 1 or y >= height - 1:
        return False

    # Get the current pixel and its neighbors
    center_pixel = image[y, x]
    neighbors = [
        image[y - 1, x],  # Above
        image[y + 1, x],  # Below
        image[y, x - 1],  # Left
        image[y, x + 1]  # Right
    ]

    # Check if any neighbor is black
    for neighbor in neighbors:
        if not np.array_equal(neighbor, target_color):
            return False

    return True


# Process a section of the image
def process_section(image, y_start, y_end, x_start, x_end):
    section = image[y_start:y_end, x_start:x_end]

    # Create a mask for pixels that should be black
    black_mask = np.zeros(section.shape[:2], dtype=bool)

    for y in prange(section.shape[0]):
        for x in prange(section.shape[1]):
            if check_neighbors(section, y, x):
                black_mask[y, x] = True

    # Apply the mask to set pixels to black
    section[black_mask] = target_color

    return section, y_start, x_start


# Load the image
image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

height, width = image.shape[:2]

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
cv2.imwrite(output_image_path, full_image)

print(f"Processing complete. Saved the modified file as {output_image_path}")

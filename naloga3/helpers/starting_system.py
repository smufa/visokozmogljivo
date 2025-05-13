import numpy as np
from PIL import Image

def create_png_with_square(width, height, filename, channel1_value, channel2_value):
    """
    Creates a two-channel PNG image with a square in the middle,
    and saves it to the specified filename using Pillow.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        filename (str): The filename to save the image to.
        channel1_value (float): The value for the first channel of the square (0.0-1.0).
        channel2_value (float): The value for the second channel of the square (0.0-1.0).
    """
    # Create an empty image with two channels (e.g., grayscale + alpha)
    img = np.zeros((height, width, 2), dtype=np.uint8)

    # Calculate the square's dimensions and position
    square_size = width // 4
    square_x = width // 2 - square_size // 2
    square_y = height // 2 - square_size // 2

    # Scale the channel values to the range 0-255
    channel1_scaled = int(channel1_value * 255)
    channel2_scaled = int(channel2_value * 255)

    # Set the square's channel values
    img[square_y:square_y + square_size, square_x:square_x + square_size, 0] = channel1_scaled
    img[square_y:square_y + square_size, square_x:square_x + square_size, 1] = channel2_scaled

    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(img, mode="LA")  # L: grayscale, A: alpha

    # Save the image as a PNG file
    img.save(filename)

if __name__ == "__main__":
    create_png_with_square(256, 256, "naloga3/data/start256.png", 0.75, 0.25)
    create_png_with_square(512, 512, "naloga3/data/start512.png", 0.75, 0.25)
    create_png_with_square(1024, 1024, "naloga3/data/start1024.png", 0.75, 0.25)
    create_png_with_square(2048, 2048, "naloga3/data/start2048.png", 0.75, 0.25)
    create_png_with_square(4096, 4096, "naloga3/data/start4096.png", 0.75, 0.25)

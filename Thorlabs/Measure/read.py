import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np

# Check if a filename was provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python your_script_name.py <filename.npy>")
    sys.exit(1) # Exit with an error code

# Get the filename from the command-line arguments
filename = sys.argv[1]

try:
    # Load the .npy file
    data = np.load(filename)

    # Print information about the array
    print(f"Successfully loaded '{filename}':")
    print("---")
    print("Array content:")
    print(data)
    print("---")
    print(f"Shape of the array: {data.shape}")
    print(f"Data type of the array: {data.dtype}")
    print("---")
	
    # Suppose `img` is your image loaded as a NumPy array
    # (H, W) for grayscale or (H, W, C) for RGB/BGR, etc.

    max_val = img.max()           # fastest, idiomatic
    # same as np.max(img) or np.amax(img)

    # If you also need the (row, col[, channel]) location:
    flat_index = np.argmax(img)              # index in the flattened array
    coords = np.unravel_index(flat_index, img.shape)

    print(f"max value = {max_val}, at index {coords}")


    # Display the array as an image
    plt.imshow(data[0])
    plt.title(f"Image from {filename}")
    plt.colorbar(label="Value") # Add a color bar for better interpretation
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Please ensure the file exists and the path is correct.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

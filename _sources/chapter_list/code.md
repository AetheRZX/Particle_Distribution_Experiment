# Code
[comment]: <> (Add code here)

## Image Processing - White
```python
# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tkinter import filedialog
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

#Select the folder
base_path = Path(filedialog.askdirectory(title='Select Folder'))

## Variables that you can change
num_pictures = 5
num_shake = 10

# Initialize dataframe and calculates the mass per area
dataframe_shake = pd.DataFrame()

# Function to check where there is tif, we will only be running the script in the directory where tif images are present
def contains_tif_files(directory_path):
    return any(file.suffix in ['.tif'] for file in directory_path.iterdir())

# Get the list of directories that contain tif images
directory_list = [d for d in base_path.iterdir() if (d).is_dir() and contains_tif_files(d)]

# Runs through each directory_list
for folder_path in directory_list:

    # Gets the directory of the tif files
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    tif_files.sort(key=lambda x: int(x.split('.')[0]))

    # Coordinate for imaging
    x_min, x_max, y_min, y_max = [200,500,500,800] 
    # x_min, x_max, y_min, y_max = [1700, 2700, 2000, 3000]

    # Stack the images into one variable to perform calculation easier
    image_stack = []
    for file in tif_files:

        image_path = folder_path / Path(file)
        image = io.imread(image_path, as_gray=True)
        image_stack.append(image)
    
    # Take the area we want to image
    image_cropped = [img[x_min:x_max, y_min:y_max] for img in image_stack[1:]]
    background_cropped = image_stack[0][x_min:x_max, y_min:y_max]

    # Perform calculation and adds it to the dataframe
    normalized = (image_cropped - background_cropped) # White on black
    average_intensity = np.mean(normalized, axis=(1,2))
    dataframe_shake["Intensity Shake"] = average_intensity /255


# Plot result
# Reshape x and y datas
x_data = np.arange(num_shake) + 1
y_data = np.reshape(dataframe_shake["Intensity Shake"], (-1,num_shake))

# Create a figure and axis
fig, ax = plt.subplots()

# Plot each set of 10 entries
for i in range(y_data.shape[0]):  # Loop through each set of 10 entries
    ax.plot(x_data, y_data[i], label=f'Set {i+1}')


# Add labels and legend
ax.set_xlabel("Trial Number")
ax.set_ylabel("Delta Intensity")
ax.grid()
ax.legend()
ax.set_xlim([1,num_shake])
ax.set_title("White Particles Intensity")
plt.show()

# Calculate the mean and standard deviation of y_data
mean_y = np.mean(y_data, axis=1)
std_y = np.std(y_data, axis=1)

# Generate set labels such as set_1, set_2, etc.
x_labels = [f'set_{i+1}' for i in range(int(len(dataframe_shake)/10))]
# Create the bar plot with error bars
plt.bar(np.arange(len(x_labels)), mean_y, yerr=std_y * 3, capsize=5, label='Mean with 3*Std Dev Error')

# Annotate the plot with the 3*std values
for i in range(len(mean_y)):
    # Dynamically adjust position of text based on the height of the bars
    height_offset = -0.0005 if mean_y[i] > 0 else -0.01  # Adjust for negative values
    plt.text(i, mean_y[i] + std_y[i] * 3 + height_offset, f'{std_y[i] * 3:.4f}', 
             ha='center', va='bottom' if mean_y[i] >= 0 else 'top')

# Add labels and legend
plt.title("Mean and Standard Deviation White Particles")
plt.grid(axis='y')
plt.xlabel("Set Number")
plt.ylabel("Mean Delta Intensity")

# Set custom x-ticks
plt.xticks(np.arange(len(x_labels)), x_labels)

# Display the plot
plt.legend()
plt.show()
```

The code appends the image into an array. This first entry of the iamge array is then the background. The background will then substract every pixel in the image array to get the difference in pixel intensity. 

The difference is then averaged over the area and plotted as a function of trial.


## SIFT
```python
import matplotlib.pyplot as plt
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage import io
from pathlib import Path
import polars as pl
import numpy as np

def main():
    # Get the directory and sort them
    base_path = Path('E:\School\Ziming\September_1\Shake')
    tif_dir = sorted(base_path.glob('*.tif'), key=lambda x: int(x.stem))

    # Read the images, we don't take the first image since we don't need the background
    images = np.array([io.imread(x).astype(float) / 255 for x in tif_dir])
    images = images[1:]

    # Make sure the trial are run 10 times
    num_images = images.shape[0]
    assert num_images % 10 == 0, "The number of images must be divisible by the number of trial (10 in this case)"

    # Reshape the array into a 4D shape: (num_groups, num_trial, x_pixels, y_pixels)
    images_grouped = images.reshape(-1, 10, images.shape[1], images.shape[2])

    # Loop trough each group
    for i in range(images_grouped.shape[0]):
        # Group the image, process them into a dataframe
        images = images_grouped[i]
        df = process_images_to_df(images)

        # Create subplots
        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(12, 18)) 
        plt.gray()

        # Loop over each subplot using ax.flat
        for j, axis in enumerate(ax.flat):
            if j < len(df):  # Ensure you don't go out of bounds
                plot_matches(axis, df['img1'][j], df['img2'][j], df['key1'][j], df['key2'][j], df['match'][j], alignment='horizontal', only_matches=True)
                axis.axis('off')
                axis.set_title(f"Tap {j + 1}")
            else:
                axis.remove()  # Remove unused subplots
        
        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.2, hspace=0.0, bottom=0.05, top=0.6) 

        # Save the figure
        plt.savefig(f'{base_path}/result/Tapped_{i+1}.png', bbox_inches = 'tight', pad_inches = 0)
        plt.show()




# Function to SIFT image, more detailed documentation can be found in scipy
def sift_image(image):
    descriptor_extractor = SIFT()
    descriptor_extractor.detect_and_extract(image)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    return keypoints1, descriptors1

def match_image(img1, img2):

    keypoints1, descriptors1 = sift_image(img1)
    keypoints2, descriptors2 = sift_image(img2)
    
    matches = match_descriptors(
    descriptors1, descriptors2, max_ratio=0.6, cross_check=True
    )

    return matches, keypoints1, keypoints2

def process_images_to_df(images):

    df = pl.DataFrame({
        "img1": pl.Series([], dtype=pl.Object),
        "img2": pl.Series([], dtype=pl.Object),
        "match": pl.Series([], dtype=pl.Object),
        "key1": pl.Series([], dtype=pl.Object),
        "key2": pl.Series([], dtype=pl.Object)
    })

    for i in range(len(images) - 1):
        
        img1 = images[0]
        img2 = images[i+1]
        match, key1, key2 = match_image(img1, img2)
        
        new_row = {
            "img1": img1,
            "img2": img2,
            "match": match,
            "key1": key1,
            "key2": key2
        }
        
        # Convert the new row to a Polars DataFrame
        new_df = pl.DataFrame([new_row])
        
        # Append the new row to the original DataFrame
        df = pl.concat([df, new_df])

    return df

if __name__ == '__main__':
    main()
```

The SIFT algorithm utilizes the scipy library. It performs a match between two arrays and outputs the matches coordinate. The matches are then plotted.

## Image Processing - Black
The code for black particles are largely the same with white. The only difference are the coordinates used for imaging as well as the delta intensity calculation. For black particles, the intensity is multiplied by -1 to get consistent data with white particles.

Below is the code

```python
# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tkinter import filedialog
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

#Select the folder
base_path = Path(filedialog.askdirectory(title='Select Folder'))

## Variables that you can change
num_pictures = 5
num_shake = 10

# Initialize dataframe and calculates the mass per area
dataframe_shake = pd.DataFrame()

# Function to check where there is tif, we will only be running the script in the directory where tif images are present
def contains_tif_files(directory_path):
    return any(file.suffix in ['.tiff'] for file in directory_path.iterdir())

# Get the list of directories that contain tif images
directory_list = [d for d in base_path.iterdir() if (d).is_dir() and contains_tif_files(d)]

# Runs through each directory_list
# Runs through each directory_list
for folder_path in directory_list:

    # Gets the directory of the tif files
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]
    tif_files.sort(key=lambda x: int(x.split('.')[0]))

    # Coordinate for imaging
    # x_min, x_max, y_min, y_max = [200,500,500,800] 
    x_min, x_max, y_min, y_max = [1700, 2700, 2000, 3000]

    # Stack the images into one variable to perform calculation easier
    image_stack = []
    for file in tif_files:

        image_path = folder_path / Path(file)
        image = io.imread(image_path, as_gray=True)
        image_stack.append(image)
    
    # Take the area we want to image
    image_cropped = [img[x_min:x_max, y_min:y_max] for img in image_stack[0:]]
    image_cropped = np.array(image_cropped)

    num_images = image_cropped.shape[0]  # Number of images
    height, width = image_cropped.shape[1], image_cropped.shape[2]  # Image dimensions

    # Calculate the number of groups of 11
    picture_per_trial = num_shake+1
    num_groups = num_images // picture_per_trial

    # Reshape the array to (num_groups, 11, height, width)
    image_reshaped = np.reshape(image_cropped[:num_groups * picture_per_trial], (num_groups, picture_per_trial, height, width))

    for i in range(len(image_reshaped)):
        background_cropped = image_reshaped[i][0]
        normalized = image_reshaped[i][1:] - background_cropped
        average_intensity = -1 * np.mean(normalized, axis=(1,2))
        normalized_df = pd.DataFrame(average_intensity, columns=["Intensity Shake"])
        dataframe_shake = pd.concat([dataframe_shake, normalized_df], ignore_index=True)

# Plot result
# Reshape x and y datas
x_data = np.arange(num_shake) + 1
y_data = np.reshape(dataframe_shake["Intensity Shake"], (-1,num_shake))

# Create a figure and axis
fig, ax = plt.subplots()

# Plot each set of 10 entries
for i in range(y_data.shape[0]):  # Loop through each set of 10 entries
    ax.plot(x_data, y_data[i], label=f'Set {i+1}')


# Add labels and legend
ax.set_xlabel("Trial Number")
ax.set_ylabel("Delta Intensity")
ax.grid()
ax.legend()
ax.set_xlim([1,num_shake])
ax.set_title("Black Particles Intensity")

# Display the plot
plt.show()

# Calculate the mean and standard deviation of y_data
mean_y = np.mean(y_data, axis=1)
std_y = np.std(y_data, axis=1)

# Generate set labels such as set_1, set_2, etc.
x_labels = [f'set_{i+1}' for i in range(int(len(dataframe_shake)/10))]

# Create the bar plot with error bars
plt.bar(np.arange(len(x_labels)), mean_y, yerr=std_y * 3, capsize=5, label='Mean with 3*Std Dev Error')

# Annotate the plot with the 3*std values
for i in range(len(mean_y)):
    # Dynamically adjust position of text based on the height of the bars
    height_offset = 0.0 if mean_y[i] > 0 else -0.02  # Adjust for negative values
    plt.text(i, mean_y[i] + std_y[i] * 3 + height_offset, f'{std_y[i] * 3:.4f}', 
             ha='center', va='bottom' if mean_y[i] >= 0 else 'top')

# Add labels and legend
plt.title("Mean and Standard Deviation Black Particles")
plt.grid(axis='y')
plt.xlabel("Set Number")
plt.ylabel("Mean Delta Intensity")

# Set custom x-ticks
plt.xticks(np.arange(len(x_labels)), x_labels)

# Display the plot
plt.legend()
plt.show()
```

## SIFT - Black
The difference with the white particles is the coordinates required for imaging.

```python
import matplotlib.pyplot as plt
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage import io
from pathlib import Path
import polars as pl
import numpy as np

def main():
    # Get the directory and sort them
    base_path = Path('E:\School\Ziming\September_11 - Black\Shake')
    tif_dir = sorted(base_path.glob('*.tif'), key=lambda x: int(x.stem))

    # Read the images, we don't take the first image since we don't need the background
    x_min, x_max, y_min, y_max = [1700, 2700, 2000, 3000]
    images = np.array([io.imread(x)[x_min:x_max, y_min:y_max].astype(float) for x in tif_dir])
    images = images[1:]

    # Make sure the trial are run 10 times
    num_images = images.shape[0]
    assert num_images % 10 == 0, "The number of images must be divisible by 10"

    # Reshape the array into a 4D shape: (num_groups, num_trial, x_pixels, y_pixels)
    images_grouped = images.reshape(-1, 10, images.shape[1], images.shape[2])

    # Loop trough each group
    for i in range(images_grouped.shape[0]):
        # Group the image, process them into a dataframe
        images = images_grouped[i]
        df = process_images_to_df(images)

        # Create subplots
        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(12, 18)) 
        plt.gray()

        # Loop over each subplot using ax.flat
        for j, axis in enumerate(ax.flat):
            if j < len(df):  # Ensure you don't go out of bounds
                plot_matches(axis, df['img1'][j], df['img2'][j], df['key1'][j], df['key2'][j], df['match'][j], alignment='horizontal', only_matches=True)
                axis.axis('off')
                axis.set_title(f"Tap {j + 1}")
            else:
                axis.remove()  # Remove unused subplots
        
        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.2, hspace=0.0, bottom=0.05, top=0.6) 

        # Save the figure
        # plt.savefig(f'{base_path}/result/Tapped_{i+1}.png', bbox_inches = 'tight', pad_inches = 0)
        plt.show()




# Function to SIFT image, more detailed documentation can be found in scipy
def sift_image(image):
    descriptor_extractor = SIFT()
    descriptor_extractor.detect_and_extract(image)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    return keypoints1, descriptors1

def match_image(img1, img2):

    keypoints1, descriptors1 = sift_image(img1)
    keypoints2, descriptors2 = sift_image(img2)
    
    matches = match_descriptors(
    descriptors1, descriptors2, max_ratio=0.6, cross_check=True
    )

    return matches, keypoints1, keypoints2

def process_images_to_df(images):

    df = pl.DataFrame({
        "img1": pl.Series([], dtype=pl.Object),
        "img2": pl.Series([], dtype=pl.Object),
        "match": pl.Series([], dtype=pl.Object),
        "key1": pl.Series([], dtype=pl.Object),
        "key2": pl.Series([], dtype=pl.Object)
    })

    for i in range(len(images) - 1):
        
        img1 = images[0]
        img2 = images[i+1]
        match, key1, key2 = match_image(img1, img2)
        
        new_row = {
            "img1": img1,
            "img2": img2,
            "match": match,
            "key1": key1,
            "key2": key2
        }
        
        # Convert the new row to a Polars DataFrame
        new_df = pl.DataFrame([new_row])
        
        # Append the new row to the original DataFrame
        df = pl.concat([df, new_df])

    return df

if __name__ == '__main__':
    main()
```
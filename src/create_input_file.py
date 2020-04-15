import os
import imageio
import numpy as np

from utils import plot_image

from skimage import color

input_folder = 'multiple_organs_photos'
output_folder = 'multiple_organs_input_files'

files = os.listdir(input_folder)

for file in files:
    image = imageio.imread(os.path.join(input_folder, file))
    image = color.rgb2gray(image)
    image[image!=0] = 1

    name = file.split('.')[0]

    file_name = "{}/{}.in".format(output_folder, name)
    np.savetxt(file_name, image)

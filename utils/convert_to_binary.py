# from PIL import Image
# im = Image.open('Mask_resized/resized_0.png')

# pixels = list(im.getdata())
# width, height = im.size
# pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
# print(pixels)

from PIL import Image
import os

# set the directory path and new size
dir_path = "./Mask_resized"
output_dir = ".\\Mask_resized_bin"
#new_size = (800, 600)  # (width, height)
import numpy as np
# loop through all files in the directory
for file_name in os.listdir(dir_path):
    # check if the file is an image
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
        # open the image file and resize it
        with Image.open(os.path.join(dir_path, file_name)) as img:
            # Convert Image to Numpy as array 
            img = np.array(img)
            print(img.shape)  
            # Put threshold to make it binary
            #binarr = np.where(img>128, 255, 0)
            binarr = img
            binarr[binarr >= 128] = 255
            binarr[binarr < 128] = 0
            # Covert numpy array back to image 
            binimg = Image.fromarray(binarr)
            new_file_name = file_name
            binimg.save(os.path.join(output_dir, new_file_name))
            
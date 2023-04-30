from PIL import Image
import os

# set the directory path and new size
dir_path = "./Mask"
output_dir = ".\\Mask_resized"
new_size = (800, 600)  # (width, height)

# loop through all files in the directory
for file_name in os.listdir(dir_path):
    # check if the file is an image
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
        # open the image file and resize it
        with Image.open(os.path.join(dir_path, file_name)) as img:
            img = img.resize(new_size)
            #print(img)
            # save the resized image with a new file name
            new_file_name = "resized_" + file_name
            #print(os.path.join(output_dir, new_file_name))
            if img.mode == 'P':
                img = img.convert('RGB')
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(os.path.join(output_dir, new_file_name))

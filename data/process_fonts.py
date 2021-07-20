import numpy as np
import pandas as pd
import string
import os
import zipfile

from PIL import Image, ImageFont, ImageDraw
from pathlib import Path


with zipfile.ZipFile('./original_fonts.zip', 'r') as zip_ref:
    zip_ref.extractall('./original_fonts')

# Transform ttf files into slightly uniformized, centered,
# black and white images (without anti-aliasing) with a specified size 

# Size of the created images:
x_image = 32
y_image = 32

os.mkdir('./fonts_32_before_selection/')

for font_path in Path("./original_fonts").iterdir():
    try:
        last_part_path = os.path.basename(os.path.normpath(font_path))
        font_images_path = "./fonts_32_before_selection/" + last_part_path
        
        os.mkdir(font_images_path[:-4])
        font = ImageFont.truetype(str(font_path), 100)

        # Uniformize a bit the size of the fonts. Some fonts do not fit in the canvas and we
        # need to remove those!
        xmax, ymax = 0, 0
        for character in string.ascii_uppercase:
            x, y = font.getsize(character)
            if x > xmax:
                xmax = x
            if y > ymax:
                ymax = y
        font = ImageFont.truetype(str(font_path), int(5000/(xmax+ymax)))

        # Create the image for each letter
        for character in string.ascii_uppercase:
            x, y = font.getsize(character)
            
            # Create black and white letter images without anti-aliasing
            image = Image.new('1', (x_image,y_image), 255)
            draw = ImageDraw.Draw(image)

            # Center the font and save images
            draw.text(((x_image - x)/2,(y_image - y)/2), character, 0, font=font)
            image.save(font_images_path[:-4] + '/' + font.getname()[0] + ' ' + \
                       font.getname()[1] + ' - ' + str(character) + '.png')

    except:
        print(last_part_path[:-4], 'could not be processed!')
        pass
    

# Some fonts were manually selected for an assortment of errors.
# The filtered data were then zipped, which we can undo using:

with zipfile.ZipFile('./fonts_32_after_selection.zip', 'r') as zip_ref:
    zip_ref.extractall('./fonts_32_after_selection')

    
# If the fonts dataframe exists in zip format, unzip it:

if Path('./fonts_32_df.zip').is_file():
    with zipfile.ZipFile('./fonts_32_df.zip', 'r') as zip_ref:
        zip_ref.extractall()


# Otherwise, cycle over the created images and create a dataframe to be used later by pytorch.
# It includes the original name of the font.

if not Path('./fonts_32_df.pkl').is_file():
    
    # Lists to put the data to be send to pandas
    fonts_arrays = []
    fonts_names = []    
    
    for i, folder in enumerate(Path('./fonts_32_after_selection').iterdir()):
        # Dictionary with the columns to be send to pandas
        dict_font = {}
        k = 0
        for character, file in zip(string.ascii_uppercase, Path(folder).iterdir()):
            img = Image.open(file)
            img_array = np.array(img)
            if k == 0 and img_array.shape == (50, 50):
                print(img_array.shape, file, )

            img_array = np.where(img_array == 255., 0., 1.).astype(np.float32)
            dict_font.update({character: img_array})
            k += 1
        fonts_arrays.append(dict_font)

        font_name = os.path.basename(os.path.normpath(file))[:-8]
        fonts_names.append(font_name)

    # Create and save dataframe
    fonts_32_df = pd.DataFrame(fonts_arrays)
    fonts_32_df['font_name'] = fonts_names
    fonts_32_df.to_pickle('fonts_32_df.pkl')
else:
    print('The dataframe with the fonts already exists!')
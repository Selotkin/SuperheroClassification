"""Split the dataset into train/val/test and resize images to 64x64.
The dataset comes into the following format:
    SuperHeroes/
        *superhero_name*_001.png
Original images have size (256, 256).
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 64
NUM_COPIES = 5

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data\\\SuperHeroes', help="Directory with the Superhero dataset")
parser.add_argument('--output_dir', default='data\\64x64_SuperHeroes', help="Where to write the new data")

        
def resize_and_save(filename, output_dir, flag, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    filename = filename.split('\\')[-1]
    if flag == 'train':
        for  i in range(NUM_COPIES):
            filename = filename.split('.')[0] + "_" + str(i) + "." + filename.split('.')[1]
            image.save(os.path.join(output_dir, filename))
    else:
        image.save(os.path.join(output_dir, filename))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, '')

    # Get the filenames in the directory
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.png')]

    # Split the images in 'train', 'val' and 'test' images into 60% train, 20% val and 20% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    
    random.seed(11)
    filenames.sort()
    random.shuffle(filenames)

    split1 = int(0.6 * len(filenames))
    split2 = int(0.8 * len(filenames))
    train_filenames = filenames[:split1]
    val_filenames = filenames[split1:split2]
    test_filenames = filenames[split2:]
    
    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print(f"Processing {split} data, saving preprocessed data to {output_dir_split}")
        for filename in tqdm(filenames[split]):            
            resize_and_save(filename, output_dir_split, split, size=SIZE)

    print("Done building dataset")
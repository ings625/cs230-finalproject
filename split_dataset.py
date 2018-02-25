"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
	train_signs/
		0_IMG_5864.jpg
		...
	test_signs/
		0_IMG_5942.jpg
		...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os
import shutil

from PIL import Image
from tqdm import tqdm


filenames1 = os.listdir('downloads/proc/train/')
filenames = [os.path.join('downloads/proc/train/', f) for f in filenames1 if f.endswith('.jpg')]
filenames1 = os.listdir('downloads/proc/dev/')
filenames += [os.path.join('downloads/proc/dev/', f) for f in filenames1 if f.endswith('.jpg')]
# test_filenames = os.listdir(test_data_dir)
# test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

# Split the images in 'train_signs' into 80% train and 20% dev
# Make sure to always shuffle with a fixed seed so that the split is reproducible
random.seed(230)
filenames.sort()
random.shuffle(filenames)

split1 = int(0.7 * len(filenames))
split2 = int(0.85 * len(filenames))
train_filenames = filenames[:split1]
dev_filenames = filenames[split1:split2]
test_filenames = filenames[split2:]

filenames = {'train': train_filenames,
			 'dev': dev_filenames,
			 'test': test_filenames}

output_dir = 'data/'

if not os.path.exists(output_dir):
	os.mkdir(output_dir)
else:
	print("Warning: output dir {} already exists".format(output_dir))

# Preprocess train, dev and test
for split in ['train', 'dev', 'test']:
	output_dir_split = os.path.join(output_dir, '{}'.format(split))
	if not os.path.exists(output_dir_split):
		os.mkdir(output_dir_split)
	else:
		print("Warning: dir {} already exists".format(output_dir_split))

	print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
	for filename in tqdm(filenames[split]):
		out_file = os.path.join(output_dir_split, filename.split('/')[-1]) 
		shutil.move(filename, out_file)

print("Done building dataset")



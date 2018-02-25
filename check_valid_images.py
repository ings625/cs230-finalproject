from PIL import Image
import os
import collections

train_filenames = ['downloads/' + f for f in os.listdir('downloads/') if f.endswith('.jpg')]
# train_labels = [int(f.split('_')[0]) for f in train_filenames]


# train_filenames = ['downloads/152_0ab9365ef4dcd475.jpg']

for file in train_filenames:

	print("Trying: " + file)

	try:
	    im=Image.open(file)
	    im.verify()
	    # do stuff
	except IOError:
	    # filename not an image file
	    print("ERROR:")
	    print(file)
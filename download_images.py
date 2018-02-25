import sys, os, multiprocessing, urllib.request, csv
from PIL import Image
import io
import collections


def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  countsfile = open('counts.csv', 'r')
  countsreader = csv.reader(countsfile)
  ids_to_take = [line[0] for line in countsreader if int(line[1]) >= 1000]
  key_url_list = [line for line in csvreader if line[2] in ids_to_take]
  print ("Num of ids: " + str(len(ids_to_take)))
  print ("Num of images: " + str(len(key_url_list)))
  return key_url_list[1:]  # Chop off header


def DownloadImage2(key_url):
  out_dir = 'downloads/'
  (key, url, landmark_id) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    response = urllib.request.urlopen(url)
    image_data = response.read()
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(StringIO(image_data))
  except:
    print('Warning: Failed to parse image %s' % key)
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print('Warning: Failed to convert image %s to RGB' % key)
    return

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    print('Warning: Failed to save image %s' % filename)
    return

def DownloadImage(key_url):
  out_dir = 'downloads/'
  (key, url, landmark_id) = key_url
  filename = os.path.join(out_dir, '%s_%s.jpg' % (landmark_id, key))

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    urllib.request.urlretrieve(url, filename=filename, reporthook=None, data=None)
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

# def Run():
#   if len(sys.argv) != 3:
#     print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
#     sys.exit(0)
#   (data_file, out_dir) = sys.argv[1:]

out_dir = 'downloads/'

if not os.path.exists(out_dir):
  os.mkdir(out_dir)

data_file = 'train.csv'

key_url_list = ParseData(data_file)


# print(key_url_list)
pool = multiprocessing.Pool(processes=50)
pool.map(DownloadImage, key_url_list)

# urllib.request.urlretrieve('http://static.panoramio.com/photos/original/70761397.jpg', filename='test.jpg', reporthook=None, data=None)

# response = urllib.request.urlopen('http://static.panoramio.com/photos/original/70761397.jpg')
# image_data = response.read()
# file = io.StringIO(str(image_data))
# pil_image = Image.open(image_data)
# pil_image_rgb = pil_image.convert('RGB')
# pil_image_rgb.save('test.jpg', format='JPEG', quality=90)


# if __name__ == '__main__':
#   Run()
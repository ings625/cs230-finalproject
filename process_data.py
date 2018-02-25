import csv
import collections

import urllib


counts = collections.defaultdict(int)

with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    i = 0
    header = ""
    for row in reader:
    	if i == 0:
    		header = row
    		i = 1
    	else:
    		counts[row[2]] += 1


with open('counts.csv', 'wb') as f:
    writer = csv.writer(f)

    for key in counts:
    	writer.writerow([key, counts[key]])    

print counts

ids_to_download = []

for key in counts:
    if counts[key] >= 1000:
        ids_to_download.append(key)


with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    i = 0
    header = ""
    for row in reader:
        if i == 0:
            header = row
            i = 1
        else:
            if 



urllib.urlretrieve("http://www.gunnerkrigg.com//comics/00000001.jpg", "00000001.jpg")
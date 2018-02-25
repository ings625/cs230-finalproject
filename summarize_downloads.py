import os
import collections
import csv

data_dir = 'data/'

train_data_dir = os.path.join(data_dir, "train")
dev_data_dir = os.path.join(data_dir, "dev")
test_data_dir = os.path.join(data_dir, "test")

# Get the filenames from the train and dev sets
train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                   if f.endswith('.jpg')]
eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                  if f.endswith('.jpg')]
test_filenames = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir)
                  if f.endswith('.jpg')]

# train_filenames = [f for f in os.listdir('downloads/') if f.endswith('.jpg')]
train_labels = [int(f.split('/')[-1].split('_')[0]) for f in train_filenames]
eval_labels = [int(f.split('/')[-1].split('_')[0]) for f in eval_filenames]
test_labels = [int(f.split('/')[-1].split('_')[0]) for f in test_filenames]
print("Training images: " + str(len(train_filenames)))
print("Dev images: " + str(len(eval_labels)))
print("Test images: " + str(len(test_labels)))


train_counts = collections.defaultdict(int)
eval_counts = collections.defaultdict(int)
test_counts = collections.defaultdict(int)

for label in train_labels:
	train_counts[label] += 1

for label in eval_labels:
	eval_counts[label] += 1

for label in test_labels:
	test_counts[label] += 1

with open('data_summary.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['landmark_id', 'num_train', 'num_dev', 'num_test'])

    for key in train_counts:
    	writer.writerow([key, train_counts[key], eval_counts[key], test_counts[key]])    


# print train_filenames[:4]
# print train_labels[:4]

# print counts

# print(len(counts))
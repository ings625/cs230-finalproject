"""Train the model"""

import argparse
import logging
import os
import random
import numpy as np

import tensorflow as tf

from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


label_mappings = {
    6051    :   0   ,
9633    :   1   ,
6599    :   2   ,
9779    :   3   ,
2061    :   4   ,
5554    :   5   ,
6651    :   6   ,
4352    :   7   ,
6696    :   8   ,
5376    :   9   ,
2743    :   10  ,
13526   :   11  ,
1553    :   12  ,
10900   :   13  ,
8063    :   14  ,
8429    :   15  ,
4987    :   16  ,
12220   :   17  ,
11784   :   18  ,
2949    :   19  ,
12718   :   20  ,
7092    :   21  ,
3804    :   22  ,
10184   :   23  ,
2338    :   24  ,
3924    :   25  ,
12172   :   26  ,
10045   :   27  ,
428 :   28  ,
10932   :   29  ,
9029    :   30  ,
1878    :   31  ,
10026   :   32  ,
3283    :   33  ,
2044    :   34  ,
3497    :   35  ,
9434    :   36  ,
7172    :   37  ,
13170   :   38  ,
5367    :   39  ,
5955    :   40  ,
7661    :   41  ,
13653   :   42  ,
6231    :   43  ,
11249   :   44  ,
2870    :   45  ,
2449    :   46  ,
11371   :   47  ,
10033   :   48  ,
1847    :   49  ,
1946    :   50  ,
10511   :   51  ,
165 :   52  ,
8169    :   53  ,
1602    :   54  ,
13444   :   55  ,
1472    :   56  ,
13873   :   57  ,
233 :   58  ,
2444    :   59  ,
13475   :   60  ,
2241    :   61  ,
12972   :   62  ,
13594   :   63  ,
878 :   64  ,
13876   :   65  ,
4300    :   66  ,
3130    :   67  ,
12676   :   68  ,
8274    :   69  ,
13332   :   70  ,
4981    :   71  ,
5460    :   72  ,
7000    :   73  ,
960 :   74  ,
2341    :   75  ,
7041    :   76  ,
9999    :   77  ,
5046    :   78  ,
7008    :   79  ,
4954    :   80  ,
6476    :   81  ,
9605    :   82  ,
5475    :   83  ,
4085    :   84  ,
10577   :   85  ,
7764    :   86  ,
152 :   87  ,
1213    :   88  ,
3426    :   89  ,
3065    :   90  ,
6597    :   91  ,
5166    :   92  ,
1546    :   93  ,
4644    :   94  ,
1310    :   95  ,
11536   :   96  ,
9296    :   97  ,
6347    :   98  ,
5421    :   99  ,
12966   :   100 ,
6846    :   101 ,
10644   :   102 ,
12647   :   103 ,
7420    :   104 ,
7840    :   105 ,
2246    :   106 ,
12965   :   107 ,
7130    :   108 ,
5206    :   109 ,
5618    :   110 ,
10496   :   111 ,
11073   :   112 ,
12937   :   113 ,
10005   :   114 ,
2145    :   115 ,
13471   :   116 ,
3034    :   117 
}


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    # overwritting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir

    np.random.seed(230)


    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                       if f.endswith('.jpg')]
    eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                      if f.endswith('.jpg')]
    # train_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
    #                    if f.endswith('.jpg') and np.random.random() <= 0.9]
    # eval_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
    #                   if f.endswith('.jpg') and np.random.random() > 0.9]


    # Labels will be between 0 and 5 included (6 classes in total)
    train_labels = [int(f.split('/')[-1].split('_')[0]) for f in train_filenames]
    train_labels_set = set(train_labels)
    # print(train_labels_set)
    train_labels = [label_mappings[t] for t in train_labels]

    eval_labels = [int(f.split('/')[-1].split('_')[0]) for f in eval_filenames]
    eval_labels_set = set(eval_labels)
    # print(eval_labels_set)
    eval_labels = [label_mappings[t] for t in eval_labels]

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
import time 
random_state =42
from dnnclassifier import DNNClassifier

from LibTools import  load_required_data
from sklearn.metrics import make_scorer
from NNTools import max_norm_regularizer, leaky_relu, get_auc_sngl_task, run_single_fit_search

### now get data
PROJECT_ROOT_DIR = "..."
target = "dep_var"
category = "binary_selected_feature"
test_features = os.path.join(PROJECT_ROOT_DIR,"data","test.csv")
train_features = os.path.join(PROJECT_ROOT_DIR,"data","train.csv")
#definr training and validation set
X, y, class_weight, unseen_X, unseen_y = load_required_data(test_features, train_features, target, weight_needed = False)
X_train, y_train, class_weight, unseen_X, unseen_y = X.values.astype(np.float32), y.values.astype(np.int32), class_weight, unseen_X.values.astype(np.float32), unseen_y.values.astype(np.int32)

#define scoring function
scoring= make_scorer(get_auc_sngl_task, greater_is_better=True, needs_proba = True) 

#Identify parameters for tarining NN
param_distribs = {
    "n_neurons": [[4000,2000,1000,1000],[4000,2000,1000,1000]],
    "batch_size": [50,100,400],
    "learning_rate": [0.01, 0.02, 0.05],
    "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01)],
    "n_hidden_layers": [ 4,  ],
    "optimizer_class": [tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer],
    "dropout_rate":[0.1,0.25,0.3,0.5],
    "weight_regularizer": [None, tf.contrib.layers.l1_regularizer(0.001)],
    "initializer": [tf.variance_scaling_initializer(),tf.contrib.layers.xavier_initializer()],
    "w_max_n_thresh": [max_norm_regularizer(threshold = 1.0),],
    "batch_norm_momentum": [None, 0.9],
    "learn_decay":[None,]    
}
#number of iteration (or number of different combination of parameters)
n_iter=15
n_epochs=1000
run_single_fit_search(X_train, y_train, unseen_X, unseen_y,param_distribs,target,category, n_iter, scoring, n_epochs)



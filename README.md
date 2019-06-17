# DL_Tabular_Classification

This Repository contains the development of deep neural network models for binary classification using a set of tabular data. 
The deep learning framework used here is `TensorFlow`.

A single task neural network [dnnclassifer.py](https://github.com/nemaminejad/DL_Tabular_Classification/blob/master/dnnclassifier.py)
is developed for binary clssification of a single target variable


## Getting Started
### Requirements
Requirements for 
`TensorFlow`
`Pandas`
`Numpy`
`Scikit-Learn`

### Installation
`git clone https://github.com/nemaminejad/DL_Tabular_Classification`

### Data
1. Collect your data, preprocess (Normalize, handle missing data,etc.)
2. Divide data into a training set, a validation set to use for finding best architecture and best performing model, a test set for final testing of model performance.
3. To use the prediction model, [dnnclassifer.py](https://github.com/nemaminejad/DL_Tabular_Classification/blob/master/dnnclassifier.py):
  - Place data in the form of `valid.csv` and `train.csv` in a `data` folder
  - Name you binary target as `dep_var`
 
4. for classification of single target variable use the example script [example_run_dnn_singletask.py](https://github.com/nemaminejad/DL_Tabular_Classification/blob/master/example_run_dnn_singletask.py)

 
### Under construction
Details of the model architectures, development and performance will be added soon.

### Author
Nastaran Emaminejad

Find me in LinkedIn: https://www.linkedin.com/in/nastaran-emaminejad-791726137/

or Twitter: https://twitter.com/N_Emaminejad
### Citation
If you found my work useful for your publications, please kindly cite this repository

"Nastaran Emaminejad, DL_Tabular_Classification, (2019), GitHub repository, https://github.com/nemaminejad/DL_Tabular_Classification"


# Machine Learning from Dataaspirant
# http://dataaspirant.com/category/machine-learning-2/

# Building Decision Tree Algorithm in Python with scikit learn
# http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/

# Balance Scale Weight & Distance Database
# http://archive.ics.uci.edu/ml/datasets/balance+scale
# 
# Data Set Information:
# 
# This data set was generated to model psychological experimental results. Each example is classified as 
# having the balance scale tip to the right, tip to the left, or be balanced. The attributes are the left weight, the 
# left distance, the right weight, and the right distance. The correct way to find the class is the greater of 
# (left-distance * left-weight) and (right-distance * right-weight). If they are equal, it is balanced.
# 
# Attribute Information:
# 
# 1. Class Name: 3 (L, B, R)
# 2. Left-Weight: 5 (1, 2, 3, 4, 5)
# 3. Left-Distance: 5 (1, 2, 3, 4, 5)
# 4. Right-Weight: 5 (1, 2, 3, 4, 5)
# 5. Right-Distance: 5 (1, 2, 3, 4, 5)

import os
import sys
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from scipy.stats import itemfreq
import itertools

def main():
#     get default project directory path
    project_directory_path = os.path.dirname(sys.argv[0])  
    
#     set file path and name
    file_path = os.path.join(project_directory_path, "balance_scale.csv")  
    
#     get balance scale data frame and show the data 
    df_balance_scale = pd.read_csv(filepath_or_buffer=file_path, sep=",")
    print("DATA FILE:")
    print(df_balance_scale)
    print()
    
#     show file information
    print("FILE INFORMATION:")
    df_balance_scale.info()
    print()
    
#    set features (labels) vector
    X = df_balance_scale.drop(labels="Class Name", axis=1)
    print("FEATURES:")
    print(X)
    print()
    
#    set target vector
    Y = df_balance_scale["Class Name"]
    print("TARGET:")
    print(Y)
    print()
    
#     get  y unique names
    Y_unique_names = list(Y.unique())
    print("TARGET UNIQUE NAME:")
    print(Y_unique_names)
    print()
    
#     split the data in train and test
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, stratify=Y)
    
#     set decision tree classifier with criterion gini index
    clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=5)
    
#     fit the classifier with traing data
    clf_gini.fit(X_train, Y_train)
    
#     export a decision tree classifier in dot format
#     generate the decision tree graph using WebGraphviz (http://www.webgraphviz.com/)
    tree.export_graphviz(clf_gini, out_file='balance_scale_tree.dot')   
    
#     get target predictions
    Y_pred = clf_gini.predict(X_test)
    print("PREDICTED TARGET FREQUENCY:")       
    Y_pred_frequency = pd.value_counts(pd.Series(Y_pred))
    print(Y_pred_frequency)
    print()
    
    print("TEST TARGET FREQUENCY:")       
    Y_test_frequency = pd.value_counts(pd.Series(Y_test))
    print(Y_test_frequency)
    print()
    
#     calculate confusion matrix
    confusion_matrix_value = confusion_matrix(Y_test, Y_pred)
    print("CONFUSION MATRIX:")
#     plot confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix_value, classes=Y_unique_names, title="Confusion Matrix")
    plt.show()
    
#     calculate and show the accuracy score
    accuracy_score_value = accuracy_score(Y_test, Y_pred) * 100
    accuracy_score_value = float("{0:.2f}".format(accuracy_score_value))
    print("ACCURACY SCORE:")
    print( "{} %".format(accuracy_score_value))
    print()
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    this function prints and plots the confusion matrix.
    normalization can be applied by setting `normalize=true`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix Without Normalization')

    print(cm)
    print()
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Test Target')
    plt.xlabel('Predicted Target')
        
if __name__ == '__main__':
    main()
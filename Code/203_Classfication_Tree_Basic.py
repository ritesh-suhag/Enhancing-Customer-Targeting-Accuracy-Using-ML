
##############################################################################
# CLASSIFICATION TREE - BASIC TEMPLATE
##############################################################################

# IMPORT REQUIRED PACKAGES 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# IMPORT SAMPLE DATA

my_df = pd.read_csv("data/sample_data_classification.csv")

# SPLIT DATA INTO INPUT AND OUTPUT OBJECTS

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# SPLIT DATA INTO TRAINING AND TEST SETS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# INSTANTIATE OUR MODEL OBJECT

clf = DecisionTreeClassifier(random_state = 42, min_samples_leaf = 7)

# TRAIN OUR MODEL

clf.fit(X_train, y_train)

# ASSESS MODEL ACCURACY

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# DEMONSTRATION OF OVER-FITTING

# Decision trees are very prone to over-fitting.

y_pred_train = clf.predict(X_train)
accuracy_score(y_train, y_pred_train)

# PLOTTING DECISION TREE - 

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)







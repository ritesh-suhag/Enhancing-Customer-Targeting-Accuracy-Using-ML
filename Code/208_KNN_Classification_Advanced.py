
##############################################################################
# KNN For Classification - ABC GROCERY TASK
##############################################################################

# ~~~~~~~~~~~~~~~~~~~ IMPORT REQUIRED PACKAGES ~~~~~~~~~~~~~~~~~~~

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORT SAMPLE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# IMPORT

# We enter rb because we are reading a file in from pickle
data_for_model = pickle.load(open("data/abc_classification_modelling.p", "rb"))

# DROP UNECESSARY COLUMNS

# Dropiing customer id as we don't really need it.
data_for_model.drop(["customer_id"], axis = 1, inplace = True)

# SHUFFLE DATA 

# It's better to shuffle the data in case we didn't know about any ordering of the data done previously.
data_for_model = shuffle(data_for_model, random_state = 42)

# Checking class balance - 
data_for_model["signup_flag"].value_counts()
# To get the percent of classes - 
data_for_model["signup_flag"].value_counts(normalize = True)
# It is a bit imbalanced but not much.


# ~~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH MISSING VALUES ~~~~~~~~~~~~~~~~~~~~~~~~~

data_for_model.isna().sum()
# Since the number of NA are very low we can directly drop them.

# Dropping missing values - 
data_for_model.dropna(how = "any", inplace = True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH OUTLIERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Even though this is a distance based model, we can argue that the outliers may not have any effect on the model.
# But the outliers may cause a bunch of observations to be bunched together when we do min max scaling.

# Dealing with outlier requires level playing field - 
outlier_investigation = data_for_model.describe()

# Based on a rough look from the describe we see potential outliers in - 
# 1. distance from store
# 2. total sales
# 3. total items
# Dealing with the same using box-plot approach from data preparation tutorial -
# We just edit the outlier columns and the name of our data frame.

outlier_columns = ["distance_from_store", "total_sales", "total_items"]

for column in outlier_columns :
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2      # Widening the factor trying not to remove too many outliers.
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[( data_for_model[column] < min_border ) | ( data_for_model[column] > max_border )].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)


# ~~~~~~~~~~~~~~~~~~~~~~ SPLIT INPUT AND OUTPUT VARIABLES ~~~~~~~~~~~~~~~~~~~~

X = data_for_model.drop("signup_flag", axis = 1)
y = data_for_model["signup_flag"]

# ~~~~~~~~~~~~~~~~~~~~~~ SPLIT OUT TRAINING AND TEST SETS ~~~~~~~~~~~~~~~~~~~~

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# ~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH CATEGORICAL VARIABLES ~~~~~~~~~~~~~~~~~~~~~

# We have a categorical variable gender. Dealing with it using the code from One_Hot_Encoding - 
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse = False, drop = "first") 

# encoder_vars_array = one_hot_encoder.fit_transform(X[categorical_vars])
# Rather than doing the above, we would run the fit method on only the train data.
# This is done because we want the model to learn from the train data and apply it to teh test data.
# This ensures rules will always be the same.
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars]) # Note - this is only transform and not fit_transform.

encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE SCALING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Instantiating the object for feature scaling.
scale_norm = MinMaxScaler()

# As scaler returns the values as an array, we can convert it to df using - 
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
# Remember to change fit_transform to transform while changing the test data.
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE SELECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Instead of using Logistic regression for feature selection, we use random forest as it performed better.
# Importing the required packages for the sam.
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 42)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features is {optimal_feature_count}.")

# To get which variables these are, we update X - 
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.grid_scores_)+1), fit.grid_scores_, marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFECV \n Optimal number of features - {optimal_feature_count}. \n At Score - {round(max(fit.grid_scores_),4)}")
plt.tight_layout()
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL ASSESSMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ASSESS MODEL ACCURACY

y_pred_class = clf.predict(X_test)

# We can get the probability instead of 0 and 1 using - 
y_pred_prob = clf.predict_proba(X_test)[:,1]

# CREATING CONFUSION MATRIX -

conf_matrix = confusion_matrix(y_test, y_pred_class)
# print(conf_matrix)

# Plot to show confusion matrix - 
plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for(i, j), corr_value in np.ndenumerate(conf_matrix) :
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

# Accuracy (the number of correct classification out of all attempted classifications)
accuracy_score(y_test, y_pred_class)

# Precision (Of all observations that were predicted as positive, how many were actually positive)
precision_score(y_test, y_pred_class)

# Recall (Of all positive observation, how many did we predict as positive)
recall_score(y_test, y_pred_class)

# F1-Score (Harmonic mean of precision and recall)
f1_score(y_test, y_pred_class)

# Although we get a bit better accuracy, we get slightly lower f-score as there is a bit imbalance between precision and recall.

# ~~~~~~~~~~~~~~~~~~~~~~~ FINDING OPTIMAL VALUE FOR K ~~~~~~~~~~~~~~~~~~~~~~~~

k_list = list(range(2,25))
accuracy_scores = []

for k in k_list:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_k_value = k_list[max_accuracy_idx]

# PLOT OF MAX DEPTH - 

plt.plot(k_list, accuracy_scores)
plt.scatter(optimal_k_value, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max K \n Optimal Value of K {optimal_k_value}, (Accuracy : {round(max_accuracy)})")
plt.xlabel("K")
plt.ylabel("Accuracy (F1 Score)")
plt.tight_layout()
plt.show()

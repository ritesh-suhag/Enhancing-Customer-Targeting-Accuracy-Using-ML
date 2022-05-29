
##############################################################################
# RANDOM FOREST FOR CLASSIFICATION - ABC GROCERY TASK
##############################################################################

# ~~~~~~~~~~~~~~~~~~~ IMPORT REQUIRED PACKAGES ~~~~~~~~~~~~~~~~~~~

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE SELECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# n_estimators is the number of trees in the forest.
# Max_features is the max number of features we use in the random sampling for trees.
clf = RandomForestClassifier(random_state = 42, n_estimators = 500, max_features = 5)
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE IMPORTANCE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This method uses the method of mean decrease in the Gini score to calculate the feature importance.
# It's not completely reliable as it inflates the importance for numeric variables.

feature_importance = pd.DataFrame(clf.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Using permutation importance - 
# It randomizes one of the variable and checks the model performance to judge the importance of the variable.
# This is many times preferred compared to the first method.

# We can specify how many times we want to apply the reshufling to a particular variable using n_repeat
result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state = 42)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variable"], permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

# We see a few negative values for gender and total_sales. This indicates that reult on shuffled data worked better than real data.
# This happens by chance and just indicates that the variables are not so important.


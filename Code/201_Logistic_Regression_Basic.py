
##############################################################################
# LOGISTIC REGRESSION BASIC TEMPELATE
##############################################################################

# IMPORT REQUIRED PACKAGES 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# IMPORT SAMPLE DATA

my_df = pd.read_csv("data/sample_data_classification.csv")

# SPLIT DATA INTO INPUT AND OUTPUT OBJECTS

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# SPLIT DATA INTO TRAINING AND TEST SETS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# INSTANTIATE OUR MODEL OBJECT

clf = LogisticRegression(random_state = 42)

# TRAIN OUR MODEL

clf.fit(X_train, y_train)

# ASSESS MODEL ACCURACY

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# We can get the probability instead of 0 and 1 using - 
y_pred_prob = clf.predict_proba(X_test)

# CREATING CONFUSION MATRIX -

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Plot to show confusion matrix - 
plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("COnfusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for(i, j), corr_value in np.ndenumerate(conf_matrix) :
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()









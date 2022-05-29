
##############################################################################
# RANDOM FOREST FOR CLASSIFICATION - BASIC TEMPLATE
##############################################################################

# IMPORT REQUIRED PACKAGES 

from sklearn.ensemble import RandomForestClassifier
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

clf = RandomForestClassifier(random_state = 42)

# TRAIN OUR MODEL

clf.fit(X_train, y_train)

# ASSESS MODEL ACCURACY

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# We can put split limiting parameter to avoid over-fitting in random forest.


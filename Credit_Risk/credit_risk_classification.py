# %%
#
# Import the modules
#

import pandas                                                 as pd
import shutil                                                 as shu
from   colorama                import Fore                    as fr
from   colorama                import Back                    as bk
from   colorama                import Style                   as st
from   pathlib                 import Path                    as pt
from   sklearn.metrics         import balanced_accuracy_score as bas
from   sklearn.metrics         import confusion_matrix        as cmx
from   sklearn.metrics         import classification_report   as csr
from   sklearn.model_selection import train_test_split        as tts
from   sklearn.linear_model    import LogisticRegression      as lre
from   imblearn.over_sampling  import RandomOverSampler       as ros
from   collections             import Counter                 as ctr

# %%
#
# Auxiliary functions
#

w, h = shu.get_terminal_size()

def printSeparator():
    print(fr.GREEN + '-' * w + fr.WHITE)
    
def printStep(stepA, stepB):
    printSeparator()
    print(fr.BLUE   + stepA)
    print(fr.YELLOW + stepB)
    printSeparator()
    
def printDFinfo(name,dfName):
    printSeparator()
    print('Name: ',name)
    printSeparator()
    print(dfName.info())    
    printSeparator()
    print('Row Count :' + fr.RED)
    print(dfName.count(),fr.WHITE)
    printSeparator()
    print(dfName.head())
    printSeparator()
    
def printReport(reportName):
    printSeparator()
    print(fr.RED,'Classification Report',fr.WHITE)
    print(reportName)
    printSeparator()
    
def printBAS(basName):
    printSeparator()
    print(fr.WHITE + 'Balanced Accuracy Score : '+ fr.RED + str(basName))
    printSeparator()

# %% [markdown]
# Step 1: Split the Data into Training and Testing Sets
# 

# %% [markdown]
# 
# Step 1.1: Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.

# %%
#
# Print program steps
#

printStep('1 - Preparation','1.1 - Read CSV file, create DF and show head'                                          )

# %%
#
# Read the CSV file from the Resources folder into a Pandas DataFrame
#

df_lending = pd.read_csv(pt('../Resources/lending_data.csv'))

#
# Review the DataFrame
#

printDFinfo('df_lending',df_lending)

# %% [markdown]
# Step 1.2: Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

# %%
#
# Print program steps
#

printStep('1 - Preparation','1.2 - Create labels X and y')

# %%
#
# Separate the data into labels and features
#
# Separate the y variable, the labels
#

y = df_lending['loan_status'];

#
# Separate the X variable, the features
#

X = df_lending.drop(columns=['loan_status']);

# %%
#
# Review the y variable Series
#

printDFinfo('y',y)

# %%
#
# Review the X variable DataFrame
#

printDFinfo('X',X)

# %% [markdown]
# Step 1.3: Check the balance of the labels variable (`y`) by using the `value_counts` function.

# %%
#
# Print program steps
#

printStep('1 - Preparation','1.3 - Check Balance of y')

# %%
#
# Check the balance of our target values
#

printSeparator()
print('total count:',fr.RED,y.count(),fr.WHITE)
print('Count by  0:',fr.RED,y.value_counts()[0],fr.WHITE)
print('Count by  1:',fr.RED,y.value_counts()[1],fr.WHITE)
print('Check      :',fr.RED,y.value_counts()[0]+y.value_counts()[1])
printSeparator()

# %% [markdown]
# Step 1.4: Split the data into training and testing datasets by using `train_test_split`.

# %%
#
# Print program steps
#

printStep('1 - Preparation ','1.4 - Train Test Split')

# %%
#
# Split the data using train_test_split
# Assign a random_state of 1 to the function
#

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)
printSeparator()
print('X_train Count          :',fr.RED,X_train.count()[0],fr.WHITE)
print('y_train Count          :',fr.RED,y_train.count(),fr.WHITE)
print('X_test  Count          :',fr.RED,X_test.count()[0],fr.WHITE)
print('y_test  Count          :',fr.RED,y_test.count(),fr.WHITE)
print('X_train + X_test Count :',fr.RED,X_train.count()[0]+X_test.count()[0],fr.WHITE)
print('y_train + y_test Count :',fr.RED,y_train.count()+y_test.count(),fr.WHITE)
printSeparator()

# %% [markdown]
# Step 2. Create a Logistic Regression Model with the Original Data

# %%
#
# Print program steps
#

printStep('2 - Logistic Regression','2.1 - Create the Logistic Regression Model')

# %% [markdown]
# Step 2.1: Fit a logistic regression model by using the training data (`X_train` and `y_train`).
# 

# %%
# 
# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
#

logistic_regression_model = lre(solver='lbfgs', random_state=1)

#
# Fit the model using training data
#

lr_model                  = logistic_regression_model.fit(X_train, y_train)

# %% [markdown]
# Step 2.2: Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

# %%
#
# Print program steps
#

printStep('2 - Logistic Regression','2.2 - Make Predictions using the Testing Data')

# %%
# 
# Make a prediction using the testing data
#

test_predictions    = logistic_regression_model.predict(X_test)
df_test_predictions = pd.DataFrame({'Predictions': test_predictions, 'Actual': y_test})
printDFinfo('df_test_predictions',df_test_predictions)

# %% [markdown]
# Step 2.3: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# %%
#
# Print program steps
#

printStep('2 - Logistic Regression','2.3 Calculate, Generate, Print metrics for the model')

# %%
#
# Print the balanced_accuracy score of the model
#

printBAS(bas(y_test, test_predictions))


# %%
# 
# Generate a confusion matrix for the model
#

cf_test_matrix = cmx(y_test, test_predictions)
printSeparator()
print('cf test matrix :',fr.RED)
print(cf_test_matrix)
printSeparator()                      

# %%
#
# Print the classification report for the model
#

testing_report = csr(y_test, test_predictions);
printReport(testing_report)


# %% [markdown]
# Step 2.4: Answer the following question.
# 
# **Question:** How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
# 
# **Answer:** `The logistic regression model was 95% accurate at predicting the healthy vs high-risk loan labels`

# %%
#
# Question answer
#

printStep('2 - Logistic Regression','2.4 - Qualify the Model')
print('The logistic regression model was'+ fr.RED +' 95% ' + fr.WHITE +'accurate at predicting the healthy vs high-risk loan labels')
printSeparator()

# %% [markdown]
# Step 3. Predict a Logistic Regression Model with Resampled Training Data

# %% [markdown]
# 
# Step 3.1: Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# %%
#
# Print program steps
#

printStep('3 - Logistic Regression Model with Resampled Training Data','3.1 - Resample the training data with the RandomOversampler')

# %%
#
# Instantiate the random oversampler model
# Assign a random_state parameter of 1 to the model
#

ros = ros(random_state=1);

#
# Fit the original training data to the random_oversampler model
#

X_ros_model, y_ros_model = ros.fit_resample(X,y)

printDFinfo('X_ros_model',X_ros_model)
printDFinfo('y_ros_model',y_ros_model)

# %%
# 
# Count the distinct values of the resampled labels data
#

printSeparator()
print('X_ros_model ',ctr(X_ros_model))
print('y_ros_model ',ctr(y_ros_model))
printSeparator()
X_ros_model.info
y_ros_model.info

# %% [markdown]
# Step 3.2: Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.

# %%
#
# Print program steps
#

printStep('3 - Logistic Regression Model with Resampled Training Data','3.2 - Train a Logistic Regression Model using the resampled data')

# %%
# 
# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
#

classifier = lre(solver='lbfgs', random_state=1)

#
# Fit the model using the resampled training data
#

classifier.fit(X_ros_model, y_ros_model)

#
# Make a prediction using the testing data
#

predictions    = classifier.predict(X_ros_model);
df_predictions = pd.DataFrame({'Predictions': predictions, 'Actual': y_ros_model});
printDFinfo('df_predictions',df_predictions)

# %% [markdown]
# Step 3.3: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# %%
#
# Print program steps
#

printStep('3 - Logistic Regression Model with Resampled Training Data','3.3 - Calculate, Generate, Print metrics for the model')

# %%
# 
# Print the balanced_accuracy score of the model
#

printBAS(bas(y_ros_model, predictions))

# %%
#
# Generate a confusion matrix for the model
#
cf_matrix = cmx(y_ros_model, predictions)
printSeparator()
print('CF Matrix :',fr.RED)
print(cf_matrix)
printSeparator()

# %%
#
# Print the classification report for the model
#

report = csr(y_ros_model, predictions)
printReport(report)

# %% [markdown]
# Step 3.4: Answer the following question

# %% [markdown]
# **Question:** How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
# 
# **Answer:** `The logistic regression model predicts the oversampled data with near-perfect accuracy (>99% accurate)`

# %%
#
# Question 2 answer
#

printStep('3 - Logistic Regression Model with Resampled Training Data','3.4 - Qualify the Model ')
print('The logistic regression model predicts the oversampled data with near-perfect accuracy (>99% accurate)')
printSeparator()



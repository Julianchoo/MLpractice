# MLpractice

## What the code does in a nutshell
It's a basic script that uses the scikit library to do a linear regression model on a dataset called "diabetes"
The 'diabetes' dataset is imported from the scikit library, is prepped (we have both the "data" and the "target" sets) and split into training and testing data. Then, the linear regression model is fitted to that data, which means it can predict what the 'target' would be given the values for the parameters in the 'data' after which we can score these predictions (probably doing it through the R2)
We then plot the real against the predicted values of target, along with the line of ideal fit (where real = predicted)

## Step by step
1. Loading and exploring the data: load the diabetes dataset and store it in a variable called 'diabetes'. Print out the shapes of the data and target sets, along with the feature names
  Looking at the dataset, we see we have 442 rows and 10 columns, where each row is probably a different person, and the columns are different datapoints about that person (age, bmi, etc.), while the target data is what our model will ultimately try to predict (some health metric related to diabetes?)
3. split the data into 4 groups: data training and testing, and target training and testing. X represents the actual data (parameters used to predict the target) and y represents the target (or whatever number we want to predict using X, presumably some measure around people's diabetes). So you'll end up with X_train, X_test, y_train and y_test. The train_test_split() function takes the data and target sets, as well as the target_size (from 0 to 1) which describes the split between training and testing data, as well as the random state (hardcoding it to any number will fix these actual splits so we can predictably replicate the results)
  What the model will do in essence is see how the data in X_train correlates with y_test, so it'll come up a function that turns a given set of values in X, to predict y. So, given a person's age, sex, bmi, etc. it'll try to predict that hea
4. Instantiate the linear regression model: the LinearRegression() is an object that passes no arguments, store it under a variable called 'model'
5. Fit the data: the .fit is a method that will do the regression over the training data to predict the values in "target"
6. Evaluate the model: apply the .score method on model, passing the X_test and y_test sets so it can see how well it's prediction does. In our case it's ending with an R2 of 43%
7. Plot the data: using the matplotlib package. First create a prediction for the values in the X_test set (the ones that **weren't** used in the model's creation) and store it under a variable called y_pred. Then use the plt.plot() passing both y_test (the 'actual') target data along with y_pred. This will generate a scatter plot of predicted values in the y axis against the actual value in the x axis
8. Create a straight line with gradient 1 (this would be the line of best fit, since it would imply actuals=predicted). numby allows us to create an array with a start, stop and nr_step arguments, so np.linspace(0,330,100) is an array that starts at 0, ends at 330 and has 100 equal jumps from 0 to 330. Store this array under a variable called x, and copy it to another variable caled y. Then, the plt.plot(x,y) will add that line to the plot you created before
9. plt.show() will create an image with your scatter plot and the line of best fit




## ChatGPT's summary
-- Copying here what GPT came up with after being prompted with "Create a readme file for this:  {the entire python script}". Sadly the formatting gets lost but the gist is there

Readme
This repository contains a Python script for performing linear regression using the scikit-learn library and visualizing the results using matplotlib.

Installation
To run the script, make sure you have the following dependencies installed:

matplotlib
numpy
scikit-learn
You can install these dependencies using pip:

Copy code
pip install matplotlib numpy scikit-learn
Usage
Import the necessary modules:
python
Copy code
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
Load and explore the dataset:
python
Copy code
diabetes = datasets.load_diabetes()
print(type(diabetes))
print(diabetes.data.shape)
print(diabetes.target.shape)
print(diabetes.feature_names)
Split the dataset into training and testing sets:
python
Copy code
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=1)
print(X_train)
Set up the linear regression model:
python
Copy code
model = LinearRegression()
Fit the model to the training data:
python
Copy code
model.fit(X_train, y_train)
Evaluate the model on the testing data:
python
Copy code
print(model.score(X_test, y_test))
Plot the predicted values against the actual values:
python
Copy code
y_pred = model.predict(X_test)
plt.plot(y_test, y_pred, ".")
Additional visualization:
python
Copy code
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()
Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License.

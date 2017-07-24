import pandas as pd
import numpy as np
import pylab as P


def loadFile(file):
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv(file, header=0)
    # Add a gender column and map it to a one or a zero
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df['EmPort'] = df['Embarked'].dropna().map( {'C' : 0, 'S': 1, 'Q': 2}).astype(int)

    # Define a 2 x 3 array
    median_ages = np.zeros((2,3))

    # For gender 0 or 1 and class 1,2,& 3
    for i in range(0, 2):
        for j in range(0, 3):
            # for slice Gender = i and PClass = j+1 drop nas and calculate the median and add it to the array
            median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()

    df['AgeFill'] = df['Age']

    # For gender(i) 0 - 1 and Pclass(j) 1-3
    for i in range(0, 2):
        for j in range(0, 3):
            # at location where age is null and gender is i and class is j set AgeFill to median_ages[gender, class]
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \
                'AgeFill'] = median_ages[i,j]


    # Drop non numeric columns
    df = df.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # Drop age because it has nulls
    df = df.drop(['Age'], axis=1)

    # Drop any rows that still have an na (which is 3)
    df = df.dropna()

    print df.head(3)

    # print df.describe()

    # Convert to a numpy array
    return df.values

train_data = loadFile('../data/train.csv')
test_data = loadFile('../data/test.csv')

# print train_data

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

print output
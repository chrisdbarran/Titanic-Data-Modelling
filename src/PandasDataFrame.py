import pandas as pd
import numpy as np
import pylab as P

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../data/train.csv', header=0)

#print "Titanic Data Types \n" , df.dtypes

# First 3 rows
#print df.head(3)

# Count of non nulls in each column and data type
#df.info()

# Calculate Mean, std, etc for every numerical column
#print df.describe()

# First ten rows of the age column
#print df['Age'][0:10]

# Type of the age column
#print type(df['Age'])

# Calculate the mean age
#print df['Age'].mean()

# And the median
#print df['Age'].median()

# Get just a list of columns
#print df[ ['Sex', 'Pclass', 'Age'] ]

# List all passengers who's age is over 60
#print df[df['Age'] > 60].describe()

# Get all passengers over 60 and return only these 4 columns
#print df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']].sort(['Sex','Pclass','Age'])

# Looking for missing values
#print  df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

# Count the number of males in each class
#for i in range(1,4):
#    print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])


# Draw a histogram of the Age profile
#df['Age'].hist()
#P.show()

# Draw a histogram of the Age profile dropping empty age values
#df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
#P.show()

# Add a gender column
#df['Gender'] = 4

# Add a gender column and map it to the first letter of the gender and capitalize it.
#df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )

# Add a gender column and map it to a one or a zero
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# Add a column based on the Embarked column

#print df['Embarked'].describe()

df['EmPort'] = df['Embarked'].dropna().map( {'Q' : 0, 'C': 1, 'S': 2}).astype(int)

# Define a 2 x 3 array
median_ages = np.zeros((2,3))

# For gender 0 or 1 and class 1,2,& 3
for i in range(0, 2):
    for j in range(0, 3):
        # for slice Gender = i and PClass = j+1 drop nas and calculate the median and add it to the array
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()



#print median_ages

df['AgeFill'] = df['Age']


# For gender(i) 0 - 1 and Pclass(j) 1-3
for i in range(0, 2):
    for j in range(0, 3):
        # at location where age is null and gender is i and class is j set AgeFill to median_ages[gender, class]
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \
                'AgeFill'] = median_ages[i,j]

# Add a new column that flags the Age as Filled
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']

# Select only the rows with empty age values and only the Gender, pclass, Age and AgeFill columns, show 10 rows
#print df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill','AgeIsNull','FamilySize']].head(10)
#print df.head(3)

#print df.describe()

# Histogram of family size.
#df['FamilySize'].hist()
#P.show()

# Tidy up for machine learning

# Drop non numeric columns
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Drop age because it has nulls
df = df.drop(['Age'], axis=1)

# Drop any rows that still have an na (which is 3)
df = df.dropna()

#print df.dtypes[df.dtypes.map(lambda x: x=='object')]

print df.describe()

# Convert to a numpy array
train_data = df.values

print train_data
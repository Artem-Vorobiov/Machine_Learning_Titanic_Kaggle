import build_model_1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def cat2num_sex(val):
    if "male" == val:
        return 1
    else:
        return 0


def cat2num_embarking(val):
    if "C" == val:
        return 2
    elif "Q" == val:
        return 1
    else:
        return 0


df = pd.read_csv("data/test.csv")

# Missing Data
df['Age'] = df['Age'].fillna(value=df.Age.median())
df['Fare'] = df['Fare'].fillna(value=df.Fare.median())


# Categorical to Numerical
df['Sex'] = df['Sex'].apply(cat2num_sex)
df['Embarked'] = df['Embarked'].apply(cat2num_embarking)

# Normalization
scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(np.array(df['Age']).reshape(-1, 1))
df['Fare'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1, 1))

# Columns
features_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X_test = np.array(df[features_cols])

print('\n\t\t\t TESTING ARRAY \n')
# print(X_test)                           #   [ [], ... , [] ]
# print('\n')
# print(type(X_test))                     #   <class 'numpy.ndarray'>
# print('\n')
# print(len(X_test))                      #   418
# print('\n')
# print(X_test.shape)                     #   (418, 7)
# print('\n')

model = build_model_1.init_model()

# load trained model
model.load_weights('models/January_18---phase1-1548480946')

# optimizer and loss
# model.compile(loss='mean_squared_error', optimizer='sgd')

# inference
predicted = model.predict(X_test)
print('\n')
print(predicted)

vals = np.round(predicted)
range = np.arange(892, 1310)

with open("output_own_3.csv", "w") as f:
    f.write("PassengerId,Survived\n")
    for x, y in zip(range, vals):
        f.write("{},{}\n".format(x, int(y[0])))
f.close()

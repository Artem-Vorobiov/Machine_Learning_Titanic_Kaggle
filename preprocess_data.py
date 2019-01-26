import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)

    df = df.sample(frac=1).reset_index(drop=True)

    # Missing Data
    df['Age'] = df['Age'].fillna(value=df.Age.median())

    # Categorical to Numerical
    df['Sex'] = df['Sex'].apply(cat2num_sex)
    df['Embarked'] = df['Embarked'].apply(cat2num_embarking)

    # Normalization
    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(np.array(df['Age']).reshape(-1, 1))
    df['Fare'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1, 1))

    # Columns
    target_cols = ["Survived"]
    features_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    y_train = df[target_cols]
    X_train = df[features_cols]

    return X_train, y_train


def wrap_preprocess():
    X_train, y_train = preprocess_data("data/train.csv")

    train_size = int(len(y_train) * 0.80)

    with h5py.File("dataset-v1.h5", 'w') as f:
        f.create_dataset("X_train", data=np.array(X_train[:train_size]))
        f.create_dataset('y_train', data=np.array(y_train[:train_size]))
        f.create_dataset("X_val", data=np.array(X_train[train_size:]))
        f.create_dataset("y_val", data=np.array(y_train[train_size:]))


wrap_preprocess()

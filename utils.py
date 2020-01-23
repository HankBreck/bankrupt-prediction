import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.utils import shuffle

def clean(file_path):
    # load data into pandas from file
    raw_data = loadarff(file_path)
    df = pd.DataFrame(raw_data[0])

    null_or_no = df.isnull()
    drop_set = set()

    # checking dataframe for null values
    for index, row in null_or_no.iterrows():
        for column in row:
            if column is True:
                drop_set.add(index)

    # drop the null values and return the dataframe
    df = df.drop(list(drop_set), axis=0)

    return df


def prepare_dataset(df):
    # returns four pandas dataframes trainX, trainY, testX, testY

    # shuffling dataframe to ensure the classes (bankrupt or non-bankrupt) are shuffled
    df = shuffle(df)
    msk = np.random.rand(len(df)) < 0.8

    # separate the dataframe into train and test and input and output
    trainY = df[msk].pop('class')
    trainX = df[msk].drop('class', axis=1)
    testY = df[~msk].pop('class')
    testX = df[~msk].drop('class', axis=1)

    return trainX, trainY, testX, testY



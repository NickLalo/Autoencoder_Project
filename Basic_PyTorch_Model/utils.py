import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
"""
place to write functions for use in other files
"""


"""
-- Scale Inputs --
Inputs should either be...
1. Normalized
- Scaled to a range between 0 and 1
If the distribution of a feature is NOT normal than it should be normalized.  You don't want to mess with the funky
distribution, but embrace and learn on it.
Will probably most of the time normalize
or 
2. Standardized
- Scaled to a mean of 0 and standard deviation of 1
If the distribution of a feature is normal (gaussian bell curve) then it should be standardized
"""


def normalize_input_features(dataframe, columnNames=[], returnScaleUserInputList=False):
    """
    Normalize between 0 and 1
    function to normalize input features of a given pandas dataframe.  When not given a list of columnNames it will
    normalize all features including categorical by converting to numbered ints and then normalizing.

    When returnScaleUserInputList is set to True it will also return an object for normalizing other input later on.  Can be useful
    for normalizing user input.
    """
    if not columnNames:  # list is empty, process all input features
        columnNames = dataframe.columns.values.tolist()[:-1]  # get column names except last column
    scaler = MinMaxScaler()  # initialize scaler object with sklearn MinMaxScaler
    scaleUserInputList = []  # setup scaleList to be populated with information for normalizing user input
    for col in columnNames:
        try:  # try to process data as numerical
            _ = int(dataframe[col].iloc[0])  # ensure that feature is float/int
            # save min and max values for processing user input
            scaleUserInputList.append(["numerical", [dataframe[col].min(), dataframe[col].max()], col])
            dataframe[[col]] = scaler.fit_transform(dataframe[[col]])  # normalize between 0 and 1
        except:  # process data that is categorical
            # change categorical data to numerical ints starting at zero
            dataframe, targetLookupTable = scale_categorical_data(dataframe=dataframe, columnNames=[col],
                                                                  returnTargetLookupTable=True)
            # reverse the targetLookupTable so it can be used to transform user input to numerical ints
            intLookupTable = {}
            for key, value in targetLookupTable.items():
                intLookupTable[value] = key
            dataframe[[col]] = scaler.fit_transform(dataframe[[col]])  # normalize
            # save lookup table for processing user input
            scaleUserInputList.append(["categorical", intLookupTable, col])
    if returnScaleUserInputList:
        return dataframe, scaleUserInputList
    return dataframe


def scale_categorical_data(dataframe, columnNames=[], returnTargetLookupTable=False):
    if not columnNames:  # list is empty, only update last col
        columnNames = [list(dataframe.columns)[-1]]
    for col in columnNames:
        uniques = dataframe[col].unique()
        targetLookupTable = {}  # only really works on last column
        for index, category in enumerate(uniques):
            # print(index, category)  # print out values
            if returnTargetLookupTable:
                targetLookupTable[index] = category
            dataframe.loc[dataframe[col] == category, col] = index
    if returnTargetLookupTable:
        return dataframe, targetLookupTable
    return dataframe


def scale_user_input(userInput, scale):
    min = scale[0]
    max = scale[1]
    if userInput < min:
        print("userInput lower than min value")
        return userInput
    if userInput > max:
        print("userInput greater than max value")

    scaledInput = (userInput - min) / (max - min)
    return scaledInput


def print_out_balance_info(dataframe):
    dataList = list(dataframe[dataframe.columns[-1]])  # get only target values
    total = 0
    for category, count in Counter(dataList).items():
        total += count

    print(f"total files: {total}")
    print(f"category\tcount\tpercent")
    for category, count in Counter(dataList).items():
        print(f"{category}\t\t\t{count}\t{count / total:.3f}%")
    return

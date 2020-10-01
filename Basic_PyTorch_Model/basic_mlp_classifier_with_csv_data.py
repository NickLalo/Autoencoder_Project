import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from basic_mlp_classifier_model import Simple_Classification_MLP
import os
import pickle
import utils

# for reproducibility which will make debugging easier.  Consider checking with different seeds?
torch.manual_seed(1337)  # comment out this line to remove seed in dataloader shuffling
random_state = 1337  # will have to modify the whole dataset shuffle line to remove reproducibility


"""
Input data is in the format...
[:-1] is the data
[-1] is the label
"""
dataDir = f"Data"
inputFilename = "binary_dataset_no_labels.csv"
# inputFilename = "unbalancedData.csv"
# inputFilename = "iris.csv"
data = pd.read_csv(f"{dataDir}{os.sep}{inputFilename}")

"""
check data for balance.  We are looking for an equal number of each type of target.
"""
utils.print_out_balance_info(dataframe=data)

"""
normalize input data and between 0 and 1
utils.normalize_input_features also turns categorical features into numbered indexes and then scales them
Returns scale List for help with normalizing user input on only numerical features
"""
data, scaleUserInputList = utils.normalize_input_features(dataframe=data, returnScaleUserInputList=True)
"""
scale outputs to indexes from zero to number of total output targets.  If the target values are far apart then the loss
function risks major penalties for incorrect answers which can lead to unstable learning.
"""
data, targetLookupTable = utils.scale_categorical_data(dataframe=data, returnTargetLookupTable=True)

# use random state for reproducibility. Delete random_state keyword argument to randomize shuffling
data = shuffle(data, random_state=random_state)

"""
Ideally we would want to split our data into test and train.  For this super basic example we will not because we need
all of the logic gate input and outputs to go through the network.  This network is not generalizing on our data but 
learning the exact formula for our specific problem.
"""
totalDataCount = data.shape[0]
test_count = int(totalDataCount * 0.1)
train_data = data[test_count:]
test_data = data[:test_count]

"""
X is the input data which will be fed into the machine learning model
y is the target label

split the input data from the target label in the dataframe using iloc[rows, columns] where rows and columns are lists
that you are picking from

In iloc[:, [-1]] the extra set of brackets around -1 ensure that pandas returns a dataframe and not a set type object
"""
train_X = train_data.iloc[:, :-1]  # all rows, all but last column
train_y = train_data.iloc[:, [-1]]  # all rows, only last column
test_X = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, [-1]]

"""
convert dataframes into tensors for PyTorch
use train_X_tensor.shape to see what these look like
"""
train_X_tensor = torch.tensor(train_X.values)
# weird formatting for dataframe with single column?
train_y_tensor = torch.tensor([npArray[0] for npArray in train_y.values])
test_X_tensor = torch.tensor(test_X.values)
test_y_tensor = torch.tensor([npArray[0] for npArray in test_y.values])

"""
Determine the input and output layer sizes
tensor.shape for the X value is [number of samples, number of attributes]
here the number of attributes is out input layer size

We have to look at the values in the y tensor to see how our model should work.
1. extract the targetValues
2. convert it to a set to get rid of duplicate values
3. get the number of values in the set
"""
inputLayerSize = train_X_tensor.shape[1]
outputLayerSize = len(set(train_y_tensor.tolist()))


class DatasetFormatter(Dataset):
    """
    class for formatting our data so it will easily be accepted by the PyTorch Dataloader.  Inherits the Dataset class
    from torch.utils.data
    """

    def __init__(self):
        self.features = ""
        self.labels = ""
        self.len = 0

    def setData(self, features, labels):
        self.features = features.float()
        self.labels = labels
        self.len = labels.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len


"""
Dataset - an object that holds the data and uses __getitem__ to return the data in the training/testing loop.  A dataset
can also hold information on how to access the data when storing all of the data in memory is not possible.  e.g. when
processing pictures the dataset would store the path to the picture and a transformation to format the picture to tensor

Dataloader - An object that is used in the training/testing loop to iterate over the data.  Can shuffle the data, set
batch size, and is configurable with num_workers for faster loading.

starting with a batch_size of 16.  Batch_size is the number of examples the model will evaluate before using the 
accumulated gradient to back propagate updating the weights and biases.
"""
batch_size = 16


trainDataset = DatasetFormatter()
trainDataset.setData(train_X_tensor, train_y_tensor)
trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

testDataset = DatasetFormatter()
testDataset.setData(test_X_tensor, test_y_tensor)  # custom made method to import data
testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True)


"""
Logic to allow our network to run on a GPU accelerated computer.  we define a bool and device, if applicable, so we can 
utilize the device for moving our network as well as input and target tensors to the GPU later on in the code.
"""
cuda_available = torch.cuda.is_available()
if cuda_available:
    device = torch.device('cuda:0')
else:
    print("cuda unavailable")

"""
Epochs - the number of times we loop over the entire training set updating the model.
learning_rate - a scalar on how much the weights and biases will update when running backprop
"""
EPOCHS = 50  # ~~~~~~~~~~~~~~~~EPOCHS~~~~~~~~~~~~~~~~
learning_rate = 0.001

"""
If cuda is available then move the model to the GPU.  This action needs to take place before initializing the optimizer. 
"""
model = Simple_Classification_MLP(inputLayer=inputLayerSize, outputLayer=outputLayerSize)
if cuda_available:
    model.cuda()
print(model)  # print out the model object to see the layers
"""
optimizer is the part of training where the model's weights and biases get update.  The optimizer determines how the 
learning rate changes over the course of the training loop.

Optimizer must be sent the model parameters.  The learning rate has a default option, however, it is worth testing out 
different values.
"""
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""
The Training Loop
"""
# TODO: create a good early stopping implementation
# TODO: save the current best weights of the model with model.state_dict() and return the best model at the end of the
#  training loop.
lossList = []
epochList = []
for epoch in range(EPOCHS):
    for data in trainDataLoader:
        # Clear accumulated gradients
        optimizer.zero_grad()

        # format data into X and y values
        X, y = data
        if cuda_available:
            X = X.cuda()
            y = y.cuda()

        # move the input data through the model and determine the output
        output = model(X)
        """
        loss function.  See how your model was wrong and determine where you can update to imrpove the accuracy
        
        Good link to investigate which loss function might be useful
        https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
        """
        # compute the overall loss by comparing the model_saves output with the target
        # lossFunction = torch.nn.NLLLoss()
        lossFunction = torch.nn.CrossEntropyLoss()
        loss = lossFunction(output, y)
        # backpropagate through the model to compute the loss for all parameters
        loss.backward()
        # step through the model and update the weights based upon the optimizers values
        optimizer.step()

    print(f"epoch:{epoch}\tloss:{round(loss.tolist(), 3)}")
    lossList.append(round(loss.tolist(), 3))
    epochList.append(epoch)

    # bad implementation of early stopping
    # if round(loss.tolist(), 3) < 0.025:
    #     break

"""
Evaluate The Model
with torch.no_grad() means our model will not remember everything that processes through it for later optimization
model.eval() will also do something similar?
"""
# check train_set accuracy
trainCorrect = 0
trainTotal = 0
with torch.no_grad():
    for data in trainDataLoader:
        X, y = data
        output = model(X)

        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                trainCorrect += 1
            trainTotal += 1
    print("Training Data Accuracy: ", round(trainCorrect / trainTotal, 3) * 100)

# check test_set accuracy
# TODO: compute confusion matrix
testCorrect = 0
testTotal = 0
with torch.no_grad():
    for data in testDataLoader:
        X, y = data
        output = model(X)

        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                testCorrect += 1
            testTotal += 1
    print("Testing Data Accuracy: ", round(testCorrect / testTotal, 3) * 100)

# test with some user input
user_input_test = True
if user_input_test:
    # getting some "user input" to run through out model.
    with open(f"{dataDir}{os.sep}{inputFilename}", "r") as infile:
        _ = infile.readline()  # read over header
        dummyData = infile.readline()  # read in user data
        dummyData = dummyData.split(",")  # format into list

        # TODO: maybe make this a function call in utils?  We will need to do this exact same code again in
        #  import_and_eval.py
        # normalize the input features
        inputFeatures = []
        for index, userInput in enumerate(dummyData[:-1]):
            inputType = scaleUserInputList[index][0]
            if inputType == "numerical":
                # scale [min, max] used to normalize user input
                scale = scaleUserInputList[index][1]
            else:  # "categorical"
                # convert from category input to integer corresponding with category
                catToIntDict = scaleUserInputList[index][1]
                userInput = catToIntDict[userInput]
                # scale [min, max] used to normalize user input
                scale = [min(list(catToIntDict.values())), max(list(catToIntDict.values()))]
            # normalize user input between 0 and 1
            newInput = utils.scale_user_input(userInput=float(userInput), scale=scale)
            inputFeatures.append(newInput)
        inputFeatures = torch.tensor(inputFeatures, dtype=torch.float)
        target = dummyData[-1]
        print(f"input to model: {inputFeatures}")
        print(f"target output: {target}")

    # torch.no_grad() tells the model to not accumulate gradients when sending data through the model.
    with torch.no_grad():
        # convert [list of values] to [[list], [of], [values]]
        inputFeatures = inputFeatures.view(-1, inputLayerSize)
        output = model(inputFeatures)
        print(f"fresh out the model: {output}")
        print(f"softmax from output {torch.softmax(output, dim=1)}")
        print(f"softmax from output summed up {torch.sum(torch.softmax(output, dim=1))}")
        print(f"highest value {round(torch.softmax(output, dim=1)[0][torch.argmax(output)].tolist(), 3)}")
        output = torch.argmax(output)  # the index of the item in the output list with the highest value
        print(f"argmax {output}")
        print(f"output of the model: {targetLookupTable[output.tolist()]}")  # converting a single value tensor to list changes it to an int?

"""
save the model so we can use it later with net.load_state_dict(torch.load(PATH))
"""
modelSaveDirectory = f"Model_Saves"
modelFilename = f"{inputFilename[:-4]}__basic_classifier_model.pth"
torch.save(model.state_dict(), f"{modelSaveDirectory}{os.sep}{modelFilename}")
pickleSaveDirectory = f"Input_Output_Scalers"
pickleFilename = f"{inputFilename[:-4]}__basic_classifier.pkl"
with open(f"{pickleSaveDirectory}{os.sep}{pickleFilename}", "wb") as outfile:
    pickle.dump(scaleUserInputList, outfile)

# TODO: redo the saving of the scaleUserInputList as a JSON because we can and pickle can sometimes be a security
#  concern

# plot out the loss curve to see how your model performed
plt.plot(epochList, lossList, color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
title = "Epoch vs Loss"
plt.title(title)
# plt.show()
plt.savefig("plot", dpi=200)

import torch.nn as nn
import torch.nn.functional as F


class Simple_Classification_MLP(nn.Module):
    def __init__(self, inputLayer=2, outputLayer=2):
        """
        Net is a basic feed forward multilayer perceptron that solves a classification problem

        inputLayer and outputLayer params are not standard for defining model_saves but helpful for this scripts adaptability
        the model parameters are defined in the init.
        nn.Linear has number of inputs, number of outputs.
            The number of weights is determined by the number of inputs
            The bias is default set to true

        The first input layer must match up with the number of features the model trains/learns on and the final output
        layer must match up with the number of target classes.

        Moving from layer one to layer two, the input of layer two must match the output of layer one.  Here fc stands
        for fully connected.
        """
        super().__init__()  # allows this class to utilize the methods from nn.Module. e.g. cuda(), parameters(), ...
        self.inputLayer = inputLayer
        self.outputLayer = outputLayer
        self.fc1 = nn.Linear(self.inputLayer, 64)  # hidden layer which the input layer goes into
        self.fc2 = nn.Linear(64, 64)  # hidden layer
        self.fc3 = nn.Linear(64, self.outputLayer)  # hidden layer which outputs to the output layer

    def forward(self, x):
        """
        forward method defines the movement from one layer to the next with model inputs x.
        In this network x moves through the first fully connected layer and evaluated by a non-linear activation
        function.  In this network we are using the relu activation function for movement in between the hidden layers
        and the softmax activation function for the output.

        activation function info:
        https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # the last activation function of a model is dependent on the loss function used.  Ensure that they match up and
        # work together.
        x = F.log_softmax(x, dim=1)  # heavily penalizes the model for a wrong prediction
        return x


if __name__ == "__main__":
    net = Simple_Classification_MLP()
    print(net)

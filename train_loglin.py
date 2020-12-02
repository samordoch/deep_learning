import loglinear as ll
import random
import utils
import numpy as np
STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    fVec=np.zeros(in_dim)
    for f in features:
        if f in utils.F2I:
            fVec[list(utils.F2I).index(f)]+=1

    print(fVec)

    return fVec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        pass
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            print("label",label)
            print("label index aka y",list(utils.L2I).index(label))
            print("features",features)
            x = feats_to_vec(features) # convert features to a vector.
            y =list(utils.L2I).index(label)                   # convert the label to number if needed.
            #i changed the parms to be induvidial
            #pn=ll.create_classifier(len(x), out_dim)
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            print("grad :",grads)

            print("===============")
            print("params :",params)
            print("===============")

            params=np.subtract(params,np.array(grads).dot(learning_rate))
            print("params after change:",params)
            print("===============")
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print (I, train_loss, train_accuracy, dev_accuracy)
    return pn

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    in_dim=len(utils.F2I)
    out_dim=len(utils.L2I)
    train_data=utils.TRAIN
    dev_data=utils.DEV
    num_iterations=30
    learning_rate=0.4
    # ...
   
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)


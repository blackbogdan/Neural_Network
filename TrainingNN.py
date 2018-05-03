
# coding: utf-8

# In[1]:


import numpy as np
import scipy.special
import scipy.misc
import scipy.ndimage
# import matplotlib.pyplot
import time
import cv2
import sys
import os

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class NeuralNetwork:
    def __init__(self):
        self.inodes = 784 # Because we have picture 28x28 pixels
        self.hnodes = 250 # We can think of this number as representation of features that correspond to 
        # picture of number. The more hidden_nodes more features our NN can see. If we choose smaller "hidden_nodes"
        # that some of features should be combined 
        self.onodes = 10 # Because we have numbers from 0 to 9
        self.lr = 0.01
        self.activation_function = lambda x: scipy.special.expit(x)
        # sys.path.append(os.path.join(os.path.dirname(__file__), 'epoch6_250hidden_nodes_rotated.npz'))
        # self.load_weights_from_file('epoch6_250hidden_nodes_rotated.npz')
        self.load_weights_from_file(os.path.join(os.path.dirname(__file__), 'epoch6_250hidden_nodes_rotated.npz'))



    def query(self, input_list):
        # convert input list to 2d array
        inputs = np.array(input_list, ndmin=2).T
        # print(inputs)
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # caclucate signals into output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def load_weights_from_file(self, filename='saved_weights.npz'):
        '''
        Loads wih and who from .npz file. Loaded values are reassigned to:
        self.wih and self.who. In this case weights would be awailable inside the class
        '''
        data = np.load(filename)
        self.wih, self.who = data['wih'], data['who']

    def predict_from_ui(self, UIinput):
        flattened = UIinput.flatten();
        inputs = (flattened / 255 * 0.99) + 0.01
        # print(inputs)
        outputs = self.query(inputs)
        print ("guess result:", outputs)
        label = np.argmax(outputs)
        return "============>>> Prediction is:", label
        # return outputs

    def predict_from_mnist(self):
        with open('mnist_test_10.csv', 'r') as f:
            test_data_list = f.readlines()
        scorecard = []
        for record in test_data_list:
            record = record.rstrip()
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
            outputs = self.query(inputs)
            label = np.argmax(outputs)
            if label == correct_label:
                scorecard.append(1)
            else: 
                scorecard.append(0)
        scorecard_array = np.asarray(scorecard)
        print('Correct guess percentage: ', scorecard_array.sum() / scorecard_array.size)
        print('Number of correctly guessed labels:', scorecard_array.sum())
        print('Size of test data: {} records'.format(scorecard_array.size))

# m = NeuralNetwork()
# m.predict_from_mnist()





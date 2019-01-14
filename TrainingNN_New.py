#!/usr/bin/env python
# coding: utf-8

# # Training Neural Network to determine number from MINST images
# 

# In[295]:


import numpy as np
import matplotlib.pyplot
import scipy.special # for activation function
import scipy.ndimage
import os
import glob
import time
# for plotting inside notebook, not in separate window
get_ipython().run_line_magic('matplotlib', 'inline')


# In[296]:


class NeuralNetwork():
    
    def __init__(self, input_nodes=784, hidden_nodes=250, output_nodes=10, learning_rate=0.01):
        self.inodes = input_nodes   # Because we have images 28x28 pixels
        self.hnodes = hidden_nodes  # We can think of this number as representation of features that correspond to 
        # picture of number. The more hidden_nodes more features our NN can see. If we choose smaller "hidden_nodes"
        # that some of features should be combined 
        self.onodes = output_nodes  # default is 10, because on images have numbers from 0 to 9
        self.lr = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)  # activation fucntion 1/(1 + e**(-x))
        self.wih = None # weights from input to hidden layers
        self.who = None # weights from hidden to output layers
        # we need to get weights, if they are present, or generate new random ones using normal distribution
        # how to get oldest file: self.chrome_extension_path = max(list_of_files, key=os.path.getctime)
        epoch_to_look_for = "epoch_0"  # Randomly created weights shall have this name
        cur_dir = os.getcwd()
        epoch_files = glob.glob(os.path.join(cur_dir, '*.npz'))
        epoch_zero_exist = any([epoch_to_look_for in file for file in epoch_files])         
        if epoch_zero_exist:
            # we have file with predefined weights, let's use it
            for file in epoch_files:
                if epoch_to_look_for in file: 
                    self.load_weights_from_file(file)
                    print("Using weights from file: {}".format(file))
        else:
            # we do not have file with weights, let's generate random weights using normal distribution:
            self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
            # Gets sample of weights from a normal probablity distribution centerd around zero
            # and with a standart deviation that is related to the nubmer of incoming links into node, 1/sqrt(number_of_nodes)
            # where 0.0 - standard deviation from normal destribution; 
            # pow(self.onodes, -0.5) 1/sqrt(number_of_nodes); 
            # (self.hnodes, self.inodes) - determines the size. F.e. (2, 3) - matrix 2 rows 3 coulumns
            self.save_weights_to_file('epoch_0_{}hidden_nodes_normal_distribution.npz'.format(hidden_nodes), 'normal_distribution')

    def load_weights_from_file(self, file_path):
        '''
        Loads wih and who from .npz file. Loaded values are reassigned to:
        self.wih and self.who. In this case weights would be available inside the class
        '''
        if file_path.endswith(".npz"):
            data = np.load(file_path)
        else:
            raise Exception("Accepted file should have .npz format. Current file: {}".format(file_path))
        self.wih, self.who = data['wih'], data['who']
    
    def save_weights_to_file(self, filename, current_precision="random"):
        '''
        Saves weights to .npz file.
        '''
        np.savez(filename, wih=self.wih, who=self.who, precision=current_precision)
        print('Saved weights to file: "{}". Specified precision: "{}"'.format(filename, current_precision))
    
    def train(self, input_list, targets_list):
        '''
        Trains nn. 
        :param: input_list - list of inputs. Should be regular python list. Must be normalized (/255 * 0.99) + 0.01) 
                before usage in this method
        :targets_list: - list of target values. Should be regular python list
        '''
        # Convert input/target lists to NP array (matrix) and transpose those. For example:
        '''
        >>> l
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> np.array(l).T
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
        >>> np.array(l, ndmin=2)
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]])
        >>> np.array(l, ndmin=2).T
        array([[ 0],
               [ 1],
               [ 2],
               [ 3],
               [ 4],
               [ 5],
               [ 6],
               [ 7],
               [ 8],
               [ 9],
               [10]])
        '''
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #  calculate signals into hidden layer. (Perform matrix multiplication)
        hidden_inputs = np.dot(self.wih, inputs)
        #  calculate signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate signals from output layer
        final_outputs = self.activation_function(final_inputs)
        
        # calculate error on the output
        output_errors = targets - final_outputs
        
        # calculate errors from hidden layer to output layer. (They shall be proportional to weights
        # from hidden layer to output layer).
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # Update weights for the links between the hidden and ouput layers (backpropagation)
                                                                                                
                                
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
        
        # Update weights for the links between input layer and hidden layer (backpropagation):
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
        
    def get_all_output_signals(self, inputs_list):
        '''
        Queries neural network with input_list
        Note, MNIST data has to be normalized (between -1 and 1, not equal to 1 and non zero)
        :input_list: should be regurar python list. In this case it shall be from MINST data set
                     Must NOT include first number(actual value on image) from MINST
        :return: list of possibilities for 10 numbers (0 to 9)
        '''
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        signals_into_output_layer = np.dot(self.who, hidden_outputs)
        # Return signals (probabilities) for numbers 
        return self.activation_function(signals_into_output_layer)
    
    def get_label_from_inputs(self, inputs_list):
        '''
        Returns label by obtaining maximum output signal.
        Inputs should be between -1 and 1, not equal to 1 and non zero
        :input_list: should be regurar python list. In this case it shall be from MINST data set. 
                     Must NOT include first number(actual value on image) from MINST
        See get_all_output_signals for more details
        '''
        return np.argmax(self.get_all_output_signals(inputs_list))
        
    def evaluate_precision_using_mnist_file(self, file_name):
        '''
        Calculates precision of current Neural Network using specified MNIST file
        Prints out the results.
        :param: file_name - name of the csv file.
        :return: current precision in %. For example, 23.59%
        '''
        with open(file_name, "r") as f:
            test_data = f.readlines()
        scorecard = []
        for record in test_data:
            record = record.rstrip()
            all_values = record.split(',')
            correct_label = int(all_values[0])
            # normalizing, so inputs will be between -1 and 1 (/255. 255 because max value is 255 in MNIST data set), 
            # not equaling to 1 (*0.99) and non zero(+0.01)
            # np.asfarray converts array with strings to array with floating point numbers
            inputs = (np.asfarray(all_values[1:])/255 * 0.99) + 0.01
            outputs = self.get_all_output_signals(inputs)
            cur_label = self.get_label_from_inputs(inputs)
            if cur_label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)
            scorecard_array = np.asarray(scorecard)
        precision_is = (scorecard_array.sum() / scorecard_array.size)*100
        print('Correct guess percentage: {}%'.format(precision_is))
        print('Size of test data: {} records'.format(scorecard_array.size))
        return precision_is
    
    def print_weights(self):
        print('====weights input hidden====')        
        print('Size: ', self.wih.size)
        print(len(self.wih[0]))
        print(self.wih[0])
        print('====weights hidden output====')
        print('Size: ', self.who[0].size)
        print(self.who[0])
    
    def load_latest_weights_from_file(self):
        '''
        Loads weights from last created file (freshest file) in currend directory
        '''
        # how to get oldest file: self.chrome_extension_path = max(list_of_files, key=os.path.getctime)
        cur_dir = os.getcwd()
        epoch_files = glob.glob(os.path.join(cur_dir, '*.npz'))
        file_name = max(epoch_files, key=os.path.getctime)
        self.load_weights_from_file(file_name)
        print("Loaded weights from file: {}".format(file_name))
    
    def rotate_inputs(self, inputs, rotation_degrees):
        '''
        Rotates current nubmer by
        :param: input_list - normalized input list
        :param: rotation_degrees - number of degrees to rotate. Negative number - rotates left
        :return: rotated input_list
        '''
        scaled_input = inputs.reshape(28, 28)
        inputs_rotated = scipy.ndimage.rotate(scaled_input, rotation_degrees, cval=0.01, order=1, reshape=False)
        # matplotlib.pyplot.imshow(inputs_rotated, cmap='Greys', interpolation='None')
        inputs_flattened = inputs_rotated.flatten()
        return inputs_flattened
    
    def train_several_epochs(self, file_name, number_of_epochs, rotation=True):
        '''
        Trains several epochs.
        1 Epoch is 1 round of training using specified file.
        Please note, this method rotates images +- 10 degrees. So technically total number of trained records is:
        number-of-records-in-file * 3.
        '''
        total_training_time = 0
        result_dict = {}
        for epoch_number in range(1, number_of_epochs + 1):
            print(">"*25, "Epoch number: ", epoch_number)
            start_epoch_time = time.time()
            with open(file_name, 'r') as f:
                # iterating through each line without loading it to memory (for example, f.readlines()
                # could cause MemoryError with large amounts of data)
                record_count = 0
                for record in f:
                    record = record.rstrip()
                    all_values = record.split(',')
                    # normalizing, so inputs will be between -1 and 1 (/255. 255 because max value is 255 in MNIST data set), 
                    # not equaling to 1 (*0.99) and non zero(+0.01)
                    inputs = (np.asfarray(all_values[1:])/255 * 0.99) + 0.01
                    # now we need the targets. all_values[0] represents actual number
                    targets = np.zeros(self.onodes) + 0.01
                    # one of targets should correspond to our number. For example, if all_values[0]=='1',
                    # then targets = [0.01, 0.99, 0.01 ... 0.01]
                    targets[int(all_values[0])] = 0.99
                    n.train(inputs, targets)
                    rotated_plus_10 = self.rotate_inputs(inputs, 10.0)     # "inputs" rotated plus 10 degress
                    n.train(rotated_plus_10, targets)
                    rotated_minus_10 = self.rotate_inputs(inputs, -10.0) # "inputs" rotated plus 10 degress
                    n.train(rotated_minus_10, targets)
                    record_count += 3 # because we trained on -10 degrees rotated, original image, +10 degrees rotation
            end_epoch_time = time.time()
            epoch_training_duration = end_epoch_time - start_epoch_time
            total_training_time += epoch_training_duration
            cur_precision = self.evaluate_precision_using_mnist_file('mnist_test.csv')
            self.save_weights_to_file('epoch_{}_{}hidden_nodes.npz'.format(epoch_number, self.hnodes), cur_precision)
            print('Epoch Training took: {} seconds'.format(epoch_training_duration))
            print('Total number of records with which we tratined NN: {}'.format(record_count))
            result_dict['epoch_{}_precision'.format(epoch_number)] = cur_precision
        print("="*80)
        print("Total duration: {} seconds".format(total_training_time))
        print("Precisions: {}".format(result_dict))
        


# In[297]:


n = NeuralNetwork()


# In[298]:


#n.print_weights()


# In[299]:


n.evaluate_precision_using_mnist_file("mnist_test.csv")


# In[300]:


n.train_several_epochs('mnist_train.csv', 10)


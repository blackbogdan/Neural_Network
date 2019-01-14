# Repository is dedicated for building Neural Network
## prerequisites
1. python3.6.5 installed and libraries:
2. numpy
3. scipy
4. maplotlib
5. jupyter

## How to run:
1. Unzip contens of the file "training_data.zip" in "Neural_Network" folder
2. jupyter notebook TrainingNN_New.ipynb
3. Go to "Cells" tab> "Run All Cells"

## What will happen after the run:
1. For each Epoch training weights shall be saved in same folder;
2. You will see logs on the bottom of jupyter notebook:
```
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  1
Correct guess percentage: 94.64%
Size of test data: 10000 records
Saved weights to file: "epoch_1_250hidden_nodes_normal_distribution.npz". Specified precision: "94.64"
Epoch Training took: 97.9292061328888 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  2
Correct guess percentage: 96.21%
Size of test data: 10000 records
Saved weights to file: "epoch_2_250hidden_nodes_normal_distribution.npz". Specified precision: "96.21"
Epoch Training took: 88.63764691352844 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  3
Correct guess percentage: 96.78999999999999%
Size of test data: 10000 records
Saved weights to file: "epoch_3_250hidden_nodes_normal_distribution.npz". Specified precision: "96.78999999999999"
Epoch Training took: 97.31674003601074 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  4
Correct guess percentage: 97.17%
Size of test data: 10000 records
Saved weights to file: "epoch_4_250hidden_nodes_normal_distribution.npz". Specified precision: "97.17"
Epoch Training took: 102.63332891464233 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  5
Correct guess percentage: 97.50999999999999%
Size of test data: 10000 records
Saved weights to file: "epoch_5_250hidden_nodes_normal_distribution.npz". Specified precision: "97.50999999999999"
Epoch Training took: 92.0697569847107 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  6
Correct guess percentage: 97.66%
Size of test data: 10000 records
Saved weights to file: "epoch_6_250hidden_nodes_normal_distribution.npz". Specified precision: "97.66"
Epoch Training took: 91.11058902740479 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  7
Correct guess percentage: 97.76%
Size of test data: 10000 records
Saved weights to file: "epoch_7_250hidden_nodes_normal_distribution.npz". Specified precision: "97.76"
Epoch Training took: 89.35434484481812 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  8
Correct guess percentage: 97.74000000000001%
Size of test data: 10000 records
Saved weights to file: "epoch_8_250hidden_nodes_normal_distribution.npz". Specified precision: "97.74000000000001"
Epoch Training took: 97.92353367805481 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  9
Correct guess percentage: 97.77%
Size of test data: 10000 records
Saved weights to file: "epoch_9_250hidden_nodes_normal_distribution.npz". Specified precision: "97.77"
Epoch Training took: 102.61093592643738 seconds
Total number of records with which we tratined NN: 180000
>>>>>>>>>>>>>>>>>>>>>>>>> Epoch number:  10
Correct guess percentage: 97.78999999999999%
Size of test data: 10000 records
Saved weights to file: "epoch_10_250hidden_nodes_normal_distribution.npz". Specified precision: "97.78999999999999"
Epoch Training took: 101.283775806427 seconds
Total number of records with which we tratined NN: 180000
================================================================================
Total duration: 960.8698582649231 seconds
Precisions: {'epoch_1_precision': 94.64, 'epoch_2_precision': 96.21, 'epoch_3_precision': 96.78999999999999, 'epoch_4_precision': 97.17, 'epoch_5_precision': 97.50999999999999, 'epoch_6_precision': 97.66, 'epoch_7_precision': 97.76,
'epoch_8_precision': 97.74000000000001, 'epoch_9_precision': 97.77, 'epoch_10_precision': 97.78999999999999}
```
Intersting fact: i was able to increase precision of epoch_10 by .31% by randomly shuffling input of the training data for each epoch.
```
Precisions: {'epoch_1_precision': 95.52000000000001, 'epoch_2_precision': 96.65, 'epoch_3_precision': 96.91, 'epoch_4_precision': 97.50999999999999, 'epoch_5_precision': 97.54, 'epoch_6_precision': 97.66, 'epoch_7_precision': 97.78,
'epoch_8_precision': 97.59, 'epoch_9_precision': 98.04, 'epoch_10_precision': 98.09}
```

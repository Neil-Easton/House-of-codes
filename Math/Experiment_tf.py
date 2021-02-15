import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

# valid_accuracy = np.load("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Optimal/valid_acc.npy")
# valid_loss = np.load("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Optimal/valid_loss.npy")

valid_accuracy = np.load("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Solution_1/valid_acc.npy")
valid_loss = np.load("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Solution_1/valid_loss.npy")

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(range(len(valid_accuracy)), valid_accuracy, 'r', label='Test Accuracy')
plt.title('Test Accuracy')
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Acc', fontsize=16)

plt.subplot(2, 1, 2)
plt.plot(range(len(valid_loss)), valid_loss, 'b', label='Test Loss')
plt.title('Test Loss')
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)

plt.show()

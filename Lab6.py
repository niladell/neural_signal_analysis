import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

data = np.load('Python_data/emu018Pdata.pkl')

## Exercise 1
cohe_data = data['cohe'].astype(np.float)
unique_cohe = sorted(set(cohe_data))
coherences = [unique_cohe[::-1],unique_cohe[1:]] #13 elements

correct = np.zeros([len(coherences)])
counter = np.zeros([len(coherences)])

for i, item in enumerate(data['correct']):
    index = coherences.index(cohe_data[i])
    if i 
    correct[index] = correct[index] + float(item)
    counter[index] = counter[index] + 1
correct = correct / counter * 100

print(correct)
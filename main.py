import numpy as np
import matplotlib.pyplot as plt
import csv

def LoadDataset(dataset):
    firstLine = True

    with open('Dataset/gender-classifier-DFE-791531.csv', newline='', encoding='Latin-1') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in reader:
            if firstLine:
                firstLine = False
                continue;

            dataset.append(row)
    return dataset;

dataset = [[]]
dataset = LoadDataset(dataset)

#ToDO:
# 1. Visualise data
# 2. Determine which columns to use
# 3. PREDICT



print("Done")


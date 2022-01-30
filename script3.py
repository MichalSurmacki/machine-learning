from os import listdir
from os.path import isfile, join
import numpy as np


files_AWE_KMC_NIE = [f for f in listdir("./scores") if isfile(join("./scores", f))]
files_OOB_UOB = [f for f in listdir("./scores_OOB_UOB") if isfile(join("./scores_OOB_UOB", f))]

scores_0_95_5 = []
scores_1_95_5 = []
scores_2_95_5 = []
scores_3_95_5 = []

scores_0_90_10 = []
scores_1_90_10 = []
scores_2_90_10 = []
scores_3_90_10 = []

scores_0_85_15 = []
scores_1_85_15 = []
scores_2_85_15 = []
scores_3_85_15 = []

counter = 0

for f in files_AWE_KMC_NIE:
    x = np.load("./scores/" + files_AWE_KMC_NIE[0])
    splited = f.split("_")
    if splited[3] == '0' and splited[4] == '0.05':
        scores_0_95_5.append(x)   
    elif splited[3] == '1' and splited[4] == '0.05':
        scores_1_95_5.append(x)   
    elif splited[3] == '2' and splited[4] == '0.05':
        scores_2_95_5.append(x)   
    elif splited[3] == '3' and splited[4] == '0.05':
        scores_3_95_5.append(x)  
    elif splited[3] == '0' and splited[4] == '0.1':
        scores_0_90_10.append(x)   
    elif splited[3] == '1' and splited[4] == '0.1':
        scores_1_90_10.append(x)
    elif splited[3] == '2' and splited[4] == '0.1':
        scores_2_90_10.append(x)
    elif splited[3] == '3' and splited[4] == '0.1':
        scores_3_90_10.append(x)
    elif splited[3] == '0' and splited[4] == '0.15':
        scores_0_85_15.append(x)
    elif splited[3] == '1' and splited[4] == '0.15':
        scores_1_85_15.append(x)
    elif splited[3] == '2' and splited[4] == '0.15':
        scores_2_85_15.append(x)
    elif splited[3] == '3' and splited[4] == '0.15':
        scores_3_85_15.append(x)

# print(len(scores_0_95_5))
# print(len(scores_1_95_5))
# print(len(scores_2_95_5))
# print(len(scores_3_95_5))
# print(len(scores_0_90_10))
# print(len(scores_1_90_10))
# print(len(scores_2_90_10))
# print(len(scores_3_90_10))
# print(len(scores_0_85_15))
# print(len(scores_1_85_15))
# print(len(scores_2_85_15))
# print(len(scores_3_85_15))



print(scores_0_95_5[0][1]])




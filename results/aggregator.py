import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import pandas as pd

listOutput = (glob.glob("*/"))

print(listOutput)

keys = []
dpath = "."
for dname in os.listdir("."):
    # print(f"Converting run {dname}",end="")
    ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
    tags = ea.Tags()['scalars']
    for tag in tags:
        keys += [tag]
    if len(keys) > 0:
        break

print(keys)
# exit(0)

listDF = []
step_sz = 50

for tb_output_folder in listOutput:
    print(tb_output_folder)
    x = EventAccumulator(path=tb_output_folder)
    x.Reload()
    x.FirstEventTimestamp()

    listValues = {}

    steps = [e.step for e in x.Scalars(keys[0])]
    # print(steps[-1])
    # wall_time = [e.wall_time for e in x.Scalars(keys[0])]
    # index = [e.index for e in x.Scalars(keys[0])]
    # count = [e.count for e in x.Scalars(keys[0])]
    n_steps = 11000
    # print(n_steps)
    # listRun = [tb_output_folder] * n_steps
    # print(listRun)
    printOutDict = {}

    data = np.zeros((n_steps, len(keys)))
    for i in range(len(keys)):
        scalars = x.Scalars(keys[i])
        for e in scalars:
            # print(e.step)
            # print(e.step, step_sz, e.value)
            data[e.step//step_sz-1, i] = e.value
        # data[:,i] = [e.value for e in x.Scalars(keys[i])]
    # exit(0)
    # print("WTF")
    #  printOutDict = {keys[0]: data[:,0], keys[1]: data[:,1],keys[2]: data[:,2],keys[3]: data[:,3]}
    printOutDict = {}
    for i in range(len(keys)):
        printOutDict[f'{keys[i]}, {tb_output_folder}'] = data[:,i]

    # printOutDict['Name'] = listRun

    DF = pd.DataFrame(data=printOutDict)
    print(DF)

    listDF.append(DF)

df = pd.concat(listDF, axis = 1)
df.index += 1
df.index *= step_sz
print(df)
df.to_csv('output2.csv')   

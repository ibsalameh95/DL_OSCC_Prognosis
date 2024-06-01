import numpy as np    


with open('prognosis_model/Data/all.txt', 'w') as all:
    all.write('# slide_id\tlabel\n')

data_arr = np.loadtxt('prognosis_model/Data/czi_filelist', delimiter='\t', comments='#', dtype=str)

slide_ids = []
labels = []
for line in data_arr:
    slide_id = line.split('/')[-1].split('.')[0]
    label = line.split('/')[-2]

    slide_ids.append(slide_id)

    if label == 'GoodProg':
        label = 0
        labels.append(0)
    if label == 'PoorProg':
        label = 1
        labels.append(1)

    with open('prognosis_model/Data/all.txt', 'a') as all:
        all.write(slide_id + '\t' + str(label) + '\n')
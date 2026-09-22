import argparse
import math
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Class label maps, shared by the dataset builders, the retrieval database and the
# morphometrics baseline. `neuron7` is the dataset used throughout the README.
LABEL7 = {'amacrine':0,'aspiny':1,'basket':2,'bipolar':3,'pyramidal':4,'spiny':5,'stellate':6}
LABEL10 = {'pyramidal':0,'aspiny':1,'cholinergic':2,'ganglion':3,'basket':4,'fast-spiking':5,'sensory':6,'neurogliaform':7,'martinotti':8,'mitral':9}

def ReadH5py(dir,normalization=True):
    with h5py.File(dir,'r') as f:
        data = f['data'][:]
        label = f['label'][:]
    if normalization:
        data = Normalization(data)
    return data,label

def Normalization(data):
    data_normalized = np.zeros(data.shape)
    for i in range(0,data.shape[0]):
        temp = data[i].copy()  # copy: otherwise the caller's array is modified in place
        origin = np.zeros((1,3))
        origin[0][0] = (temp[:,0].max() + temp[:,0].min()) / 2
        origin[0][1] = (temp[:,1].max() + temp[:,1].min()) / 2
        origin[0][2] = (temp[:,2].max() + temp[:,2].min()) / 2
        temp[:,0] = temp[:,0] - origin[0][0]
        temp[:,1] = temp[:,1] - origin[0][1]
        temp[:,2] = temp[:,2] - origin[0][2]
        if temp[:,0].max()>1:
            temp[:,0] = temp[:,0] / temp[:,0].max()
        if temp[:, 1].max() > 1:
            temp[:,1] = temp[:,1] / temp[:,1].max()
        if temp[:, 2].max() > 1:
            temp[:,2] = temp[:,2] / temp[:,2].max()
        data_normalized[i] = temp
    return data_normalized

def WriteH5py(dir,data,label):
    with h5py.File(dir,'w') as f:
        f['data'] = data
        f['label'] = label


def ReadSWC(dir,thresold,CLIP=False,Padding=True):
    data = []
    with open(dir,'r') as f:
        for line in f:
            if line[0] == '#'or line[0] == '\n':
                continue
            _,_,x,y,z,_,_ = [float(v) for v in line.split()]
            data.append([x,y,z])
    if Padding:
        while len(data)<thresold:data.append([0,0,0])
    if CLIP:
        length = math.floor(len(data) / thresold)
        data = np.array(data[0:(length*thresold)])
    else:data = np.array(data)
    return data

def GenerateH5py(dir_list,thresold):
    '''Stack every .swc in one class directory. Returns (None,None) if it holds none.'''
    chunks = []
    for filename in os.listdir(dir_list):
        if filename.split('.')[-1] != 'swc':continue
        print(dir_list.split('/')[-1],'/',filename,' ',len(chunks))
        points = ReadSWC(dir_list+'/'+filename,thresold)
        if points.shape[0]<thresold:
            continue
        chunks.append(points)
    if not chunks:
        return None,None
    datas = np.concatenate(chunks).reshape(-1,thresold,3)
    labels = np.ones((datas.shape[0], 1))*int(LABEL7[dir_list.split('/')[-1]])
    return datas,labels

def GenerateNeuronDataset(neuron_list,thresold,proportion):
    data_chunks = []
    label_chunks = []
    for neuron_type in os.listdir(neuron_list):
        data,label = GenerateH5py(neuron_list+'/'+neuron_type,thresold)
        if data is None:
            continue
        data_chunks.append(data)
        label_chunks.append(label)
    datas = np.concatenate(data_chunks)
    labels = np.concatenate(label_chunks)
    print(datas.shape,' ',labels.shape)
    state = np.random.get_state()
    np.random.shuffle(datas)
    np.random.set_state(state)
    np.random.shuffle(labels)
    datas = datas.astype(np.float32)
    labels = labels.astype(np.uint8)
    WriteH5py(dir = r'./TrainDatasets_6000.h5',data=datas[0:math.ceil(proportion*datas.shape[0])],
              label=labels[0:math.ceil(proportion*labels.shape[0])])
    WriteH5py(dir=r'./TestDatasets_6000.h5', data=datas[math.ceil(proportion * datas.shape[0]):-1],
              label=labels[math.ceil(proportion * labels.shape[0]):-1])


def VisualizeH5py(dir):
    datas,labels = ReadH5py(dir,normalization=False)
    datas2,labels2 = ReadH5py(dir,normalization=True)
    fig = plt.figure(dpi=180)
    ax1 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(0,200):
        data = datas[i]
        data2 = datas2[i]
        ax1.cla()
        ax1.scatter(data[:,0],data[:,1],data[:,2],c="b", marker=".", s=15, linewidths=0, alpha=1)
        ax2.cla()
        ax2.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c="r", marker=".", s=15, linewidths=0, alpha=1)
        plt.pause(0.5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate morphological dataset')
    parser.add_argument('--swc_dir',type=str,default='./neuron7',help='file path of .swc files')
    args = parser.parse_args()
    GenerateNeuronDataset(neuron_list=args.swc_dir,thresold=6000,proportion=0.7)

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import shutil
import torchvision.datasets as dset
import numpy as np
import scipy.io.wavfile as wav
import random as rd
import pickle as pk
import cv2
from pydub import AudioSegment
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#####################################################################
song_name = "和田光司 - Butter-Fly"
bpm = 82.46
#####################################################################
name = song_name + ".wav"
note_name = song_name + ".json"
note = json.loads(open(note_name,encoding = "utf-8").read())
song = AudioSegment.from_wav(name)
song_time = song.duration_seconds
total = song_time * bpm * 8
dead = (song_time - 5) * bpm * 8
_, signal = wav.read(name)
group = int(len(signal)/6400)

base = "test/0"
os.makedirs(base)
for i in range(group):
    pic = np.zeros((80,80,3),dtype='uint8')
    for j in range(80):
        for k in range(80):
            pic[j][k][0] = round((signal[i*6400+j*80+k][0]+32768) / 65535 * 255) 
            pic[j][k][1] = round((signal[i*6400+j*80+k][1]+32768) / 65535 * 255)
    cv2.imwrite(base + "/" + str(i) + ".jpg",pic)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])
test_augs = transforms.Compose([
transforms.Resize(size=256),
transforms.CenterCrop(size=224),
transforms.ToTensor(),
normalize
])

def predict(data_iter, net, device=None):
    result = []
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            output = net(X.to(device))
            result.append(int(output.argmax(dim=1)))
    return result

def ty_predict(data_iter, net, device=None):
    result = []
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            output = net(X.to(device))
            if output[0][2] >= -1:
                output[0][2] = 1.5
            result.append(int(output.argmax(dim=1)))
    return result

def exist_predict(data_iter, net, device=None):
    result = []
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            output = net(X.to(device))
            if output[0][0] < 1:
                output[0][0] = 0
            result.append(int(output.argmax(dim=1)))
    return result

test_imgs = ImageFolder("test", transform=test_augs)
test_iter = DataLoader(test_imgs, 1, shuffle=False)
ty = ty_predict(test_iter, torch.load("TypeModel.pt",map_location=torch.device('cpu')))
time = predict(test_iter, torch.load("TimeModel.pt",map_location=torch.device('cpu')))
exist = exist_predict(test_iter, torch.load("ExistModel.pt",map_location=torch.device('cpu')))
pos = predict(test_iter, torch.load("PosModel.pt",map_location=torch.device('cpu')))

for i in range(group):
    if ty[i] == 0:
        for j in range(1,3):
            ty[min(i+j,group-1)] = 3
            exist[min(i+j,group-1)] = 1
        exist[i] = 1

note_list = []
count = 0
for i in range(group):
    if exist[i] == 0:
        continue
    tick = round(i*total/(group-1))
    if tick >= dead:
        break
    a = {'hold_tick': 0, 
         'type': 0, 
         'next_id': 0, 
         'x': 0.05*pos[i], 
         'tick': tick, 
         'page_index': int(tick / 960), 
         'id': count, 
         'has_sibling': False, 
         'is_forward': False}
    if a['page_index'] == 0:
        continue
    if a['x'] == 0.65 and a['page_index'] % 2 == 0:
        a['x'] = 0.35
    if a['x'] == 0.05 and a['page_index'] % 2 == 0:
        a['x'] = 0.95
    if a['x'] == 0.7 and a['page_index'] % 2 == 0:
        a['x'] = 0.3
    if ty[i] == 1:
        a['type'] = 0
    elif ty[i] == 2:
        a['hold_tick'] = time[i] * bpm
        if a['page_index'] == int((a['tick'] + a['hold_tick']) / 960):
            a['type'] = 1
        else:
            a['type'] = 2
    elif ty[i] == 0:
        a['type'] = 3
        a['next_id'] = count + 1
    elif ty[i] == 3:
        a['type'] = 4
        if i + 1 >= group or ty[i+1] != 3:
            a['next_id'] = -1
        else:
            a['next_id'] = count + 1
        if (a['x'] >= 0.5) != (note_list[-1]['x'] >= 0.5):
            a['x'] = 1 - a['x'] 
    try:
        if ty[i-1] == 2 and a['x'] == note_list[-1]['x']:
            a['x'] = 1 - a['x']
    except:
        pass
    note_list.append(a)
    count += 1
    if exist[i] == 2:
        a['x'] = 1 - a['x']
        a['id'] = count
        note_list.append(a)
        count += 1
note['note_list'] = note_list
json.dump(note,open(note_name,"w",encoding = "utf-8"))

shutil.rmtree("test")

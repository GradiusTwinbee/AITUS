import numpy as np
import scipy.io.wavfile as wav
import random as rd
import pickle as pk
import torch
import cv2
import os
from pydub import AudioSegment
base = "C:/Users/123/Desktop/AITUS/"

for i in range(1,51):
    _, signal = wav.read('C:/Users/123/Desktop/AITUS/Music/'+str(i)+'.wav')
    length = len(signal)
    group = int(length/6400)
    song = AudioSegment.from_wav('C:/Users/123/Desktop/AITUS/Music/'+str(i)+'.wav')
    time = song.duration_seconds
    chain_id = []
    ty = [''] * group
    num = [0] * group
    pos = [-1] * group
    last = [-1] * group
    note = open('C:/Users/123/Desktop/AITUS/HardNote/'+str(i)+'_hard.txt')
    for line in note:
        if line[0]!='L':
            continue
        line = line.split(' ')
        line.pop(0)
        line.pop()
        for item in line:
            chain_id.append(int(item))
    chain_id.sort()
    chain_id.append(-1)
    note = open('C:/Users/123/Desktop/AITUS/HardNote/'+str(i)+'_hard.txt')
    for line in note:
        if line[0]!='N':
            continue
        line = line.split('\t')
        no = round(float(line[2])*(group-1)/time)
        pos[no] = round(float(line[3])/0.05)
        num[no] = min(num[no]+1,2)
        value = float(line[4])
        if int(line[1]) == chain_id[0]:
            chain_id.pop(0)
            ty[no] = 'chain'
        elif value == 0:
            ty[no] = 'click'
        else:
            ty[no] = 'hold'
            last[no] = max(min(round(value/0.1),9),1)
    for j in range(group):
        pic = np.zeros((80,80,3),dtype='uint8')
        for k in range(80):
            for l in range(80):
                pic[k][l][0] = round((signal[j*6400+k*80+l][0]+32768) / 65535 * 255) 
                pic[k][l][1] = round((signal[j*6400+k*80+l][1]+32768) / 65535 * 255)  
        if rd.random()>0.8:
            pat = "test/"
        else:
            pat = "train/"
        name = "/" + str(i) + "_" + str(j) + ".jpg"
        cv2.imwrite(base + "NumData/" + pat + str(num[j]) + name,pic)
        if num[j] == 0:
            continue
        cv2.imwrite(base + "PosData/" + pat + str(pos[j]) + name,pic)
        cv2.imwrite(base + "TypeData/" + pat + ty[j] + name,pic)
        if last[j] == -1:
            continue
        cv2.imwrite(base + "TimeData/" + pat + str(last[j]) + name,pic)    
    print(i)

from konlpy.tag import Mecab
from collections import Counter 
import torch
import numpy as np

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def pos(snt):
    m = Mecab()
    sts = m.pos(snt)

    lst = []
    for pos in sts:
        pos = pos[1].split('+')
        for cat in pos:
            lst.append(cat[0])

    cnt = Counter(lst)
    fin = np.array([cnt['N'], cnt['V'], cnt['M'], cnt['I'], cnt['J'], cnt['E'], cnt['X']]) 
    fin = softmax(fin)
    # if torch
    # fin = torch.Tensor(fin)
    
    return print(fin)

pos(input())
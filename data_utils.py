import numpy as np
from torch import Tensor
import librosa
from pathlib import Path
from torch.utils.data import Dataset
from RawBoost import  process_Rawboost_feature	
from utils import pad


def _audio_path(base_dir, utt_id):
    base_dir = Path(base_dir)
    candidates = (
        base_dir / f"{utt_id}.flac",
        base_dir / "flac" / f"{utt_id}.flac",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Audio for {utt_id} not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            #_, key, _, _, _ = line.strip().split()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list

class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=66800      
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(_audio_path(self.base_dir, utt_id), sr=16000)
        Y=process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad= pad(Y, self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target

class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800 # take ~4 sec audio 
        self.track = track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(_audio_path(self.base_dir, utt_id), sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id  

class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + utt_id, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id

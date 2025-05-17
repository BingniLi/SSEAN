import numpy as np, itertools, random, copy, math

import torch
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import pickle
import pandas as pd
import numpy as np
import json
import sys
from tqdm import tqdm

DEFAULT_PAD = "[PAD]"

def ad_tl(x, target_length, padding_value=0):
    current_length = x.size(0)
    if current_length > target_length:
        return x[:target_length]
    elif current_length < target_length:
        padding_length = target_length - current_length
        return torch.cat((x, torch.full((padding_length,), padding_value)))
    else:
        return x

class IEMOCAPDataset(Dataset):
    
    def __init__(self,split):

        self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.videoAudio, self.uu, self.sk, self.lk, self.nu, self.trainIds, self.testIds, self.validIds, self.videoSpeakers, self.videoLabels, self.videoLabels, self.keys\
        = pickle.load(open('iemocap/iemocap_features_all.pkl', 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
            
        self.len = len(self.keys)
        
        
    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.from_numpy(self.roberta1[vid]),  torch.from_numpy(self.roberta2[vid]), torch.from_numpy(self.roberta3[vid]), torch.from_numpy(self.roberta4[vid]),\
               torch.from_numpy(self.videoAudio[vid]), \
               torch.from_numpy(self.uu[vid]), torch.from_numpy(self.sk[vid]), torch.from_numpy(self.lk[vid]), torch.from_numpy(self.nu[vid]),\
               [[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]], \
               [1] * len(self.videoLabels[vid]), \
               self.videoLabels[vid],\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        def tensor_convert(x):
            if isinstance(x, torch.Tensor):
                return x.clone().detach()
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            else:
                return torch.tensor(x, dtype=torch.float32)
        return [pad_sequence([tensor_convert(x) for x in dat[i]]) if i < 8 else pad_sequence([torch.tensor(x) for x in dat[i]], True) if i < 10 else dat[i].tolist() for i in range(len(dat.columns))]
    
class Dataset_I(Dataset):
    
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, _, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        
        self.speakers, self.labels, \
        self.new_dialogues, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('iemocap/iemocap_kb_roberta.pkl', 'rb'), encoding='latin1')
        
        _,_, self.roberta1, self.roberta2, self.roberta3, self.roberta4,_,_,_,_ = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'))
        
        all_ids = self.trainIds + self.validIds + self.testIds
        
        roberta = RobertaModel.from_pretrained(
            'roberta.large',
            checkpoint_file='model.pt',
            data_name_or_path='iemocap-bin'
        )
        roberta.eval()
        
        uu, sk, lk, nu = {}, {}, {}, {}
        for k in tqdm(range(len(all_ids))):
            item = all_ids[k]
            newuttrs = self.new_dialogues[k]
            uu[item]= []
            sk[item]= []
            lk[item]= []
            nu[item]= []
            for uttr in newuttrs:
                if len(uttr) > 3:
                    uu[item].append(ad_tl(roberta.encode(uttr[0]),target_length=100))
                    sk[item].append(ad_tl(roberta.encode(uttr[1]),target_length=100))
                    lk[item].append(ad_tl(roberta.encode(uttr[2]),target_length=100))
                    nu[item].append(ad_tl(roberta.encode(uttr[3]),target_length=100))
                else:
                    uu[item].append(ad_tl(roberta.encode(uttr[0]),target_length=100))
                    sk[item].append(ad_tl(roberta.encode(DEFAULT_PAD), target_length=100))
                    lk[item].append(ad_tl(roberta.encode(DEFAULT_PAD), target_length=100))
                    nu[item].append(ad_tl(roberta.encode(uttr[1]),target_length=100))
                    print(f"Warning: uttr length less than 3, skipping item: {item},{uttr[0],uttr[1]}")
               
            uu[item]= pad_sequence(uu[item], batch_first=True).float()
            sk[item]= pad_sequence(sk[item], batch_first=True).float()
            lk[item]= pad_sequence(lk[item], batch_first=True).float()
            nu[item]= pad_sequence(nu[item], batch_first=True).float()
            
        self.uu = uu
        self.sk = sk
        self.lk = lk
        self.nu = nu
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        del self.videoAudio['Ses05F_script02_2']
        del self.videoSpeakers['Ses05F_script02_2']
        del self.videoLabels['Ses05F_script02_2']
        del self.uu['Ses05F_script02_2']
        del self.sk['Ses05F_script02_2']
        del self.lk['Ses05F_script02_2']
        del self.nu['Ses05F_script02_2']
        self.testIds.remove('Ses05F_script02_2')
        self.testVid.remove('Ses05F_script02_2')
        
        self.keys = all_ids

        self.len = len(self.keys)

        # 归一化
        # Loss 0.6901 F1-score 72.11
        # for vid in self.trainVid:
        #     self.videoAudio[vid] = (self.videoAudio[vid] - np.mean(self.videoAudio[vid])) / np.std(self.videoAudio[vid])
        #     self.uu[vid] = (self.uu[vid] - self.uu[vid].mean()) / self.uu[vid].std()
        #     self.sk[vid] = (self.sk[vid] - self.sk[vid].mean()) / self.sk[vid].std()
        #     self.lk[vid] = (self.lk[vid] - self.lk[vid].mean()) / self.lk[vid].std()
        #     self.nu[vid] = (self.nu[vid] - self.nu[vid].mean()) / self.nu[vid].std()
        # for vid in self.testVid:
        #     self.videoAudio[vid] = (self.videoAudio[vid] - np.mean(self.videoAudio[vid])) / np.std(self.videoAudio[vid])
        #     self.uu[vid] = (self.uu[vid] - self.uu[vid].mean()) / self.uu[vid].std()
        #     self.sk[vid] = (self.sk[vid] - self.sk[vid].mean()) / self.sk[vid].std()
        #     self.lk[vid] = (self.lk[vid] - self.lk[vid].mean()) / self.lk[vid].std()
        #     self.nu[vid] = (self.nu[vid] - self.nu[vid].mean()) / self.nu[vid].std()
            
        pickle.dump([self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.videoAudio, self.uu, self.sk, self.lk, self.nu,  self.videoSpeakers, self.videoLabels, self.videoLabels, self.trainIds, self.testIds, self.validIds], open('iemocap/iemocap_features_all.pkl', 'wb'))

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),  torch.FloatTensor(self.roberta2[vid]), torch.FloatTensor(self.roberta3[vid]), torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.uu[vid]), torch.FloatTensor(self.sk[vid]), torch.FloatTensor(self.lk[vid]), torch.FloatTensor(self.nu[vid]),\
               [[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]], \
               [1] * len(self.videoLabels[vid]), \
               self.videoLabels[vid],\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence([torch.tensor(x) for x in dat[i]]) if i < 10 else pad_sequence([torch.tensor(x) for x in dat[i]], True) if i < 12 else dat[i].tolist() for i in range(len(dat.columns))]


class MELDDataset(Dataset):
    
    def __init__(self, classify, split):
        
        if classify == 3:
            self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.videoAudio, self.uu, self.sk, self.lk, self.nu, self.trainIds, self.testIds, self.validIds, self.Speakers, self.videoLabels, self.videoLabels,self.keys\
        = pickle.load(open('meld/meld_features_all_3.pkl', 'rb'), encoding='latin1')
        else :
            self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.videoAudio, self.uu, self.sk, self.lk, self.nu, self.trainIds, self.testIds, self.validIds, self.Speakers, self.videoLabels, self.videoLabels,self.keys\
        = pickle.load(open('meld/meld_features_all_7.pkl', 'rb'), encoding='latin1')
            
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        
        self.len = len(self.keys)
        
    def __getitem__(self, index):
        vid = self.keys[index]
        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.clone().detach()
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).float()
            elif isinstance(data, list):
                return torch.from_numpy(np.array(data, dtype=np.float32))
            else:
                return torch.tensor(data, dtype=torch.float32)
            
        return (to_tensor(self.roberta1[vid]),  to_tensor(self.roberta2[vid]), to_tensor(self.roberta3[vid]),to_tensor(self.roberta4[vid]),\
               to_tensor(self.videoAudio[vid]), \
               to_tensor(self.uu[vid]), to_tensor(self.sk[vid]), to_tensor(self.lk[vid]), to_tensor(self.nu[vid]),\
               torch.FloatTensor(self.Speakers[vid]), \
               [1] * len(self.videoLabels[vid]), \
               self.videoLabels[vid],\
               vid)

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        def tensor_convert(x):
            if isinstance(x, torch.Tensor):
                return x.clone().detach()
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                return torch.tensor(x)
        return [pad_sequence([tensor_convert(x) for x in dat[i]]) if i < 10 else pad_sequence([tensor_convert(x) for x in dat[i]], True) if i < 12 else dat[i].tolist() for i in range(len(dat.columns))]


class Dataset_M(Dataset):

    def __init__(self, path, classify, train=True, valid=False):
        self.videoIDs, self.videoSpeakers, self.emotion_labels, self.videoText, \
        self.videoAudio, self.videoSentence, self.trainVid, \
        self.testVid, self.sentiment_labels = pickle.load(open(path, 'rb'))
        
        _,_,_, self.roberta1, self.roberta2, self.roberta3, self.roberta4,_,_,_,_ = pickle.load(open('meld/meld_features_roberta.pkl', 'rb'))

        if classify == 'emotion':
            self.videoLabels = self.emotion_labels
        else:
            self.videoLabels = self.sentiment_labels
        '''
        emotion_label_mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        setiment_label_mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
        '''
        
        self.speakers, self.emotion_labels, self.sentiment_labels,  \
        self.new_dialogues, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('meld/meld_kb_roberta.pkl', 'rb'), encoding='latin1')
        
        all_ids = self.trainIds + self.validIds + self.testIds
        
        roberta = RobertaModel.from_pretrained(
            'roberta.large',
            checkpoint_file='model.pt',
            data_name_or_path='meld-bin'
        )
        roberta.eval()
        
        uu, sk, lk, nu = {}, {}, {}, {}
        for k in tqdm(range(len(all_ids))):
            item = all_ids[k]
            newuttrs = self.new_dialogues[k]
            uu[item]= []
            sk[item]= []
            lk[item]= []
            nu[item]= []
            for uttr in newuttrs:
                if len(uttr) > 3:
                    uu[item].append(ad_tl(roberta.encode(uttr[0]),target_length=500))
                    sk[item].append(ad_tl(roberta.encode(uttr[1]),target_length=500))
                    lk[item].append(ad_tl(roberta.encode(uttr[2]),target_length=500))
                    nu[item].append(ad_tl(roberta.encode(uttr[3]),target_length=500))
                else:
                    uu[item].append(ad_tl(roberta.encode(uttr[0]),target_length=500))
                    sk[item].append(ad_tl(roberta.encode(DEFAULT_PAD), target_length=500))
                    lk[item].append(ad_tl(roberta.encode(DEFAULT_PAD), target_length=500))
                    nu[item].append(ad_tl(roberta.encode(uttr[1]),target_length=500))
                    print(f"Warning: uttr length less than 3, skipping item: {item},{uttr[0],uttr[1]}")
            uu[item]= pad_sequence(uu[item], batch_first=True).float()
            sk[item]= pad_sequence(sk[item], batch_first=True).float()
            lk[item]= pad_sequence(lk[item], batch_first=True).float()
            nu[item]= pad_sequence(nu[item], batch_first=True).float()
            
        self.uu = uu
        self.sk = sk
        self.lk = lk
        self.nu = nu
        
        del self.videoAudio[1432]
        del self.videoSpeakers[1432]
        del self.videoLabels[1432]
        del self.uu[1432]
        del self.sk[1432]
        del self.lk[1432]
        del self.nu[1432]
        del self.roberta1[1432]
        del self.roberta2[1432]
        del self.roberta3[1432]
        del self.roberta4[1432]
        self.testIds.remove(1432)
        
        self.keys = all_ids
        self.len = len(self.keys)
        
        if classify == 'emotion':
            pickle.dump([self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.videoAudio, self.uu, self.sk, self.lk, self.nu, self.trainIds, self.testIds, self.validIds, self.videoSpeakers, self.videoLabels, self.videoLabels,self.keys], open('meld/meld_features_all_7.pkl', 'wb'))
        else:
            pickle.dump([self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.videoAudio, self.uu, self.sk, self.lk, self.nu, self.trainIds, self.testIds, self.validIds, self.videoSpeakers, self.videoLabels, self.videoLabels,self.keys], open('meld/meld_features_all_3.pkl', 'wb'))
            
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),  torch.FloatTensor(self.roberta2[vid]), torch.FloatTensor(self.roberta3[vid]), torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.uu[vid]), torch.FloatTensor(self.sk[vid]), torch.FloatTensor(self.lk[vid]), torch.FloatTensor(self.nu[vid]),\
               torch.FloatTensor(self.speakers[vid]), \
               [1] * len(self.videoLabels[vid]), \
               self.videoLabels[vid],\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence([torch.tensor(x) for x in dat[i]]) if i < 13 else pad_sequence([torch.tensor(x) for x in dat[i]], True) if i < 15 else dat[i].tolist() for i in range(len(dat.columns))]

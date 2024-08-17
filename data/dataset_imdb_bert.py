import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import json
from bert.tokenization_bert import BertTokenizer

import h5py
from refer.imdb import IMDB
import os.path as osp

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


class ImdbDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror', 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music', 'Animation', 'Musical', 'Western', 'Sport', 'Short', 'Film-Noir']
        self.image_transforms = image_transforms
        self.split = split

        self.max_tokens = 512
        self.DATA_DIR = 'mmimdb/'
        self.IMAGE_DIR = osp.join(self.DATA_DIR, 'image')
        self.JSON_DIR = osp.join(self.DATA_DIR, 'json')
        img_list = os.listdir(self.IMAGE_DIR)
        # json_list = os.listdir(self.JSON_DIR)
        self.data_list = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        with open('mmimdb/split.json', 'r') as f:
            split_list = json.load(f)
        if self.split == 'train':
            split_list = split_list['train']
        elif self.split == 'dev':
            split_list = split_list['dev']
        elif self.split == 'test':
            split_list = split_list['test']
            
        for sample_id in split_list:
            img_name = '{}.jpeg'.format(sample_id)
            img_path_list = [osp.join(self.IMAGE_DIR, img_name)]
            json_path = osp.join(self.JSON_DIR, '{}.json'.format(img_name.replace('.jpeg', '')))
            with open(json_path, 'r') as f:
                text_dict = json.load(f)
            # text = text_dict['title'] + '[SEP]'+ text_dict['plot'][0]
            title = text_dict['title'] if 'title' in text_dict else ''
            plot = text_dict['plot'][0] if 'plot' in text_dict else ''
            content = title + ' [SEP] ' + plot
            content = content[:self.max_tokens]
            attention_mask = [0] * self.max_tokens
            padded_input_ids = [0] * self.max_tokens
            input_ids = self.tokenizer.encode(text=content, add_special_tokens=True)

            # truncation of tokens
            input_ids = input_ids[:self.max_tokens]

            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1]*len(input_ids)

            # assert len(text_dict['plot']) == 1
            label_list = text_dict["genres"]
            gt_label = np.zeros(len(self.classes), dtype=np.float32)
            for label in label_list:
                try:
                    idx = self.classes.index(label)
                    gt_label[idx] = 1
                except:
                    print(label)
            # gt_label = int(label_dict[sample_id - 1]) - 1
            info = dict(img_path_list=img_path_list, content=content, input_ids=torch.tensor(padded_input_ids).unsqueeze(0), attention_mask=torch.tensor(attention_mask).unsqueeze(0), gt_label=gt_label)
            self.data_list.append(info)

        # self.eval_mode = eval_mode
        

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        
        this_img = data['img_path_list'][0]
        gt_label = data['gt_label']
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        
        img = Image.open(this_img).convert("RGB")

        # if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
        img_1, img_2 = self.image_transforms(img, img)

        return img_1, gt_label,  input_ids, attention_mask

import os
import json
import random
from PIL import Image
import srsly

import torch
from torch.utils.data import Dataset
from data.utils_nwpu import pre_question

from torchvision.datasets.utils import download_url

class vqa_tr_dataset(Dataset):
    def __init__(self, transform, image_root, ann_root, split="nwpu_train", max_words=30, prompt=''):
        self.split = split        

        self.transform = transform
        self.image_root = image_root

        filenames = {'nwpu_train':'nwpu_train_ctq.json','nwpu_val':'nwpu_val_ctq.json', 
                     'kvqg_train':'kvqg_train_ctq.json','kvqg_val':'kvqg_val_ctq.json',
                     'textrs_train':'textrs_train_ctq.json','textrs_val':'textrs_val_ctq.json'}  # nwpu_val_gt_1k  nwpu_val.json
        file_name = os.path.join(ann_root,filenames[split])
        if not os.path.isfile(file_name):
            print('Unzipping dataset')
            file_data = srsly.read_gzip_json(file_name+'.gz')
            with open(file_name, "w") as outfile:
                json.dump(file_data, outfile)
        self.annotation = json.load(open(file_name,'r'))

        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1 
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        if 'nwpu' in self.split:
            image_path = os.path.join(os.path.join(self.image_root,ann['category']),ann['image'])   
        else:
            image_path = os.path.join(self.image_root,ann['image'])   
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split in ['nwpu_val', 'kvqg_val', 'textrs_val']:
            triplet = pre_question(ann['ann_trsent'])   
            question_id = ann['id']       
            img_id = ann['image_id']    
            caption = pre_question(ann['caption'])       
            return image, triplet, question_id, int(img_id), caption   


        elif self.split in ['nwpu_train', 'kvqg_train', 'textrs_train']:                       
            caption = pre_question(ann['caption'])       
            question = pre_question(ann['question'])        
            triplets = [pre_question(ann['ann_trsent'])]

            return image, caption, question, triplets, self.img_ids[ann['image_id']] 
        
        
def vqa_tr_collate_fn(batch):
    image_list, caption_list, question_list, triplet_list, n, imgid_list = [], [], [], [], [], []
    for image, caption, question, triplet, img_id in batch:
        image_list.append(image)
        caption_list.append(caption)     
        question_list.append(question)   
        triplet_list += triplet
        n.append(len(triplet))
        imgid_list.append(img_id)
    return torch.stack(image_list,dim=0), caption_list, question_list, triplet_list, n, imgid_list        
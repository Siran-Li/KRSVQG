import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class nwpu_caption(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, prompt=''):        
        ''' 
        image_root (string): Root directory of images (e.g. nwpu/images/)
        ann_root (string): directory to store the annotation file
        '''        
        filenames = {'train':'nwpu_train.json','val':'nwpu_val.json', 'test': 'nwpu_test.json'}  # nwpu_val_gt_1k  nwpu_val.json
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
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
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
class nwpu_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. nwpu/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'train':'nwpu_train.json','val':'nwpu_val.json', 'test': 'nwpu_test.json'}  #nwpu_val_gt_1k  nwpu_val.json
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
         
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image_id']
        
        return image, int(img_id)    
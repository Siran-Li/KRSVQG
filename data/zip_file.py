import srsly
import json
import os
from tqdm import tqdm

# files = {'nwpu-300':['nwpu_train_ctq.json', 'nwpu_val_ctq.json', 'nwpu_val_ctq_gt.json']}
# image_root = '/data/siran/RSVQG/NWPU-Captions/NWPU-RESISC45/'

# files = {'nwpu-caption':['nwpu_train.json', 'nwpu_val.json', 'nwpu_test.json']}
# image_root = '/data/siran/RSVQG/NWPU-Captions/NWPU-RESISC45/'

files = {'nwpu-caption':['nwpu_val_gt.json']}
image_root = '/data/siran/RSVQG/NWPU-Captions/NWPU-RESISC45/'

# files = {'textrs-300':['textrs_train_ctq.json', 'textrs_val_ctq.json', 'textrs_val_ctq_gt.json']}
# image_root = '/data/siran/RSVQG/TextRS/images/'

# files = {'kvqg':['kvqg_train_ctq.json', 'kvqg_val_ctq.json', 'kvqg_val_ctq_gt.json']}
# image_root = '/data/siran/RSVQG/KVQG/kvqg_images/'

for path, file_list in files.items():
    for file_name in file_list:
        with open(f'{path}/{file_name}') as user_file:
            data = json.load(user_file)
        # data = srsly.read_gzip_json(f"{path}/{file_name}.gz")
        for i in tqdm(range(len(data))):
            if '_ctq_gt' in file_name or 'val_gt' in file_name:
                if image_root in data['annotations'][i]['image']:
                    data['annotations'][i]['image'] = data['annotations'][i]['image'].replace(image_root, '')
                    if 'nwpu' in path:
                        data['annotations'][i]['category'] = data['annotations'][i]['image'].split('/')[0]
                        data['annotations'][i]['image'] = data['annotations'][i]['image'].replace(data['annotations'][i]['category']+'/', '')
                if image_root in data['annotations'][i]['image']:
                    raise Exception(data['annotations'][i]['image'])
            
            else:
                if image_root in data[i]['image']:
                    data[i]['image'] = data[i]['image'].replace(image_root, '')
                    if 'nwpu' in path:
                        data[i]['category'] = data[i]['image'].split('/')[0]
                        data[i]['image'] = data[i]['image'].replace(data[i]['category']+'/', '')
                if image_root in data[i]['image']:
                    raise Exception(data[i]['image'])
                
        # with open(f'{path}/{file_name}') as user_file:
        #     data = json.load(user_file)
        # new_path = path.replace('-json','')
        # os.makedirs(new_path, exist_ok=True) 
        srsly.write_gzip_json(f"{path}/{file_name}.gz", data)
import re
import json
import os
import srsly

import torch
import torch.distributed as dist

import utils

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    # dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file



from pycocotools.coco import COCO
from evaluate.eval import COCOEvalCap

def rs_caption_eval(coco_gt_root, results_file, split):
    
    filenames = {'val':'nwpu_val_gt.json', 'nwpu_val': 'nwpu_val_ctq_gt.json',
                 'kvqg_val':'kvqg_val_ctq_gt.json', 'textrs_val': 'textrs_val_ctq_gt.json'}    
    
    # unzip datasets
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    if not os.path.isfile(annotation_file):
        print('Unzipping dataset')
        file_data = srsly.read_gzip_json(annotation_file+'.gz')
        with open(annotation_file, "w") as outfile:
            json.dump(file_data, outfile)
    
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco=coco, cocoRes=coco_result, content='caption')

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.4f}')

    return coco_eval

def rs_question_eval(coco_gt_root, results_file, split):
    
    filenames = {'nwpu_val': 'nwpu_val_ctq_gt.json', 'kvqg_val':'kvqg_val_ctq_gt.json', 'textrs_val': 'textrs_val_ctq_gt.json'}    
    
    # unzip datasets
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    if not os.path.isfile(annotation_file):
        print('Unzipping dataset')
        file_data = srsly.read_gzip_json(annotation_file+'.gz')
        with open(annotation_file, "w") as outfile:
            json.dump(file_data, outfile)

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco=coco, cocoRes=coco_result, content='question')

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    print('Question Eval: ')
    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.4f}')
    
    return coco_eval
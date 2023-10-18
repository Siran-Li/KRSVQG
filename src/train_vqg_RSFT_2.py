'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqg_c2q import blip_vqg_c2q
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_tr_dataset import vqa_tr_collate_fn
from data.utils_nwpu import save_result, rs_caption_eval, rs_question_eval


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cg', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_qg', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    for i,(image, caption, question, triplet, n, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)

        loss, loss_cg, loss_qg = model(image, triplet, caption, question, train=True, n=n)        
        
        optimizer.zero_grad()
        loss_qg.backward()
        optimizer.step()    
        
        metric_logger.update(loss_cg=loss_cg.item())
        metric_logger.update(loss_qg=loss_qg.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQG test result:'
    print_freq = 50
    
    result = []
        
    get_img = []
    for n, (image, triplets, question_id, image_id, gt_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             

        captions, questions = model(image, triplets, caption=gt_caption, train=False) 
        
        for caption, triplet, question, img_id in zip(captions, triplets, questions, image_id):
            if img_id not in get_img:
                get_img.append(img_id)
                result.append({"image_id": img_id.item(), "caption":caption, "triplet": triplet, "question":question})             

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating vqa datasets")
    if config['dataset'] == 'nwpu':
        datasets = create_dataset('vqa_nwpu', config)
    elif config['dataset'] == 'textrs':
        datasets = create_dataset('vqa_textrs', config)  
    else:
        raise Exception('Missing dataset name')  
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_tr_collate_fn,None]) 
    #### Model #### 
    print("Creating model")
    if args.evaluate:
        model = blip_vqg_c2q(pretrained=os.path.join(args.output_dir, 'checkpoint_best.pth'), 
                            pre_cgmodel=config['pre_cgmodel'], freeze_img=args.freeze_img,
                            image_size=config['image_size'],  vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                            vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'], qg_weight=config['qg_weight'])
    else:
        model = blip_vqg_c2q(pretrained=config['pretrained'], pre_cgmodel=config['pre_cgmodel'], freeze_img=args.freeze_img,
                          image_size=config['image_size'],  vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                          vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'], qg_weight=config['qg_weight'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    config['init_lr'] = float(config['init_lr'])
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0 
       
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device) 

        else:         
            break               

        vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
        val_result_file = save_result(vqa_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')  

        if utils.is_main_process():   
            if config['dataset'] == 'nwpu':
                coco_val = rs_caption_eval(config['ann_root'],val_result_file,'nwpu_val')
                coco_val_qg = rs_question_eval(config['ann_root'],val_result_file,'nwpu_val')
            elif config['dataset'] == 'textrs':
                coco_val = rs_caption_eval(config['ann_root'],val_result_file,'textrs_val')
                coco_val_qg = rs_question_eval(config['ann_root'],val_result_file,'textrs_val')
            else:
                raise Exception('Missing dataset name')   
      
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] + coco_val_qg.eval['CIDEr'] + coco_val_qg.eval['Bleu_4'] > best:
                best = coco_val_qg.eval['CIDEr'] + coco_val_qg.eval['Bleu_4'] + coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                best_epoch = epoch  
    
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_cg_{k}': v for k, v in coco_val.eval.items()},
                        **{f'val_qg_{k}': v for k, v in coco_val_qg.eval.items()},                           
                        'epoch': epoch,
                        'best_epoch': best_epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")      

    if args.evaluate:
        vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
        val_result_file = save_result(vqa_result, args.result_dir, 'val_results', remove_duplicate='image_id')  
        
        if config['dataset'] == 'nwpu':
                coco_val = rs_caption_eval(config['ann_root'],val_result_file,'nwpu_val')
                coco_val_qg = rs_question_eval(config['ann_root'],val_result_file,'nwpu_val')
        elif config['dataset'] == 'textrs':
            coco_val = rs_caption_eval(config['ann_root'],val_result_file,'textrs_val')
            coco_val_qg = rs_question_eval(config['ann_root'],val_result_file,'textrs_val')
        else:
            raise Exception('Missing dataset name')   
        
        log_stats = {**{f'val_cg_{k}': v for k, v in coco_val.eval.items()},
                            **{f'val_qg_{k}': v for k, v in coco_val_qg.eval.items()},                       
                            }
        with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")   
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqg_kvqg_nwpu_2.yaml') 
    parser.add_argument('--output_dir', default='output/NWPU-300/Model4')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=41, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--freeze_img', action='store_true', default=False)     
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
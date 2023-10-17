from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np

class BLIP_VQG_C2Q_QG(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med2_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,  
                 prompt = 'a picture of ',                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()  
        
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        
        decoder_config = BertConfig.from_json_file(med_config)        
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        self.text_decoder_cg = BertLMHeadModel(config=encoder_config)   

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1       


    def forward(self, image, triplet, caption=None, question=None, n=None, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        triplet = self.tokenizer(triplet, padding='longest', truncation=True, max_length=35, 
                                return_tensors="pt").to(image.device) 
        triplet.input_ids[:,0] = self.tokenizer.enc_token_id 

        if train:               
            '''
            n: number of answers for each question
            weights: weight for each answer
            ''' 

            text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)     
            text.input_ids[:,0] = self.tokenizer.bos_token_id
            
            decoder_targets_cg = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
            decoder_targets_cg[:,:self.prompt_length] = -100
        
            decoder_output_cg = self.text_decoder_cg(text.input_ids, 
                                            attention_mask = text.attention_mask, 
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,                  
                                            labels = decoder_targets_cg,
                                            return_dict = True,   
                                            )   
            # loss_cg = decoder_output_cg.loss    

            cap_embeds = decoder_output_cg.hidden_states[-1]   
            cap_atts = torch.ones(cap_embeds.size()[:-1],dtype=torch.long).to(image.device) 

            question = self.tokenizer(question, padding='longest', return_tensors="pt").to(image.device) 
            question.input_ids[:,0] = self.tokenizer.bos_token_id
            question_targets = question.input_ids.masked_fill(question.input_ids == self.tokenizer.pad_token_id, -100)      

            triplet_output = self.text_encoder(triplet.input_ids, 
                                                attention_mask = triplet.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    

            triplet_states = []                
            triplet_atts = []  
            for b, n in enumerate(n):
                triplet_states += [triplet_output.last_hidden_state[b]]*n
                triplet_atts += [triplet.attention_mask[b]]*n                
            triplet_states = torch.stack(triplet_states,0)    
            triplet_atts = torch.stack(triplet_atts,0)    

            captr_embeds = torch.cat((cap_embeds, triplet_states), dim=1)
            captr_atts = torch.cat((cap_atts, triplet_atts), dim=1)

            question_output = self.text_decoder(question.input_ids, 
                                              attention_mask = question.attention_mask, 
                                              encoder_hidden_states = captr_embeds,
                                              encoder_attention_mask = captr_atts,                  
                                              labels = question_targets,
                                              return_dict = True,   
                                              reduction = 'none',
                                             )      
            
            loss_qg = question_output.loss
            loss_qg = loss_qg.sum()/image.size(0)

            return loss_qg
            

        else: 
            num_beams=3

            triplet_output = self.text_encoder(triplet.input_ids, 
                                                attention_mask = triplet.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True)
            
            text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)     
            text.input_ids[:,0] = self.tokenizer.bos_token_id
            
            decoder_targets_cg = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
            decoder_targets_cg[:,:self.prompt_length] = -100
            decoder_output_cg = self.text_decoder_cg(text.input_ids, 
                                                    attention_mask = text.attention_mask, 
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,                  
                                                    labels = decoder_targets_cg,
                                                    return_dict = True,   
                                                    )   

            prompt = [self.prompt] * image.size(0)
            input_ids_cg = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
            input_ids_cg[:,0] = self.tokenizer.bos_token_id
            input_ids_cg = input_ids_cg[:, :-1] 

            triplet_states = triplet_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
            triplet_atts = torch.ones(triplet_states.size()[:-1],dtype=torch.long).to(triplet_states.device)

            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
            outputs_cg = self.text_decoder_cg.generate(input_ids=input_ids_cg,
                                                        max_length=30,
                                                        min_length=10,
                                                        num_beams=num_beams,
                                                        eos_token_id=self.tokenizer.sep_token_id,
                                                        pad_token_id=self.tokenizer.pad_token_id,     
                                                        repetition_penalty=1.0,
                                                        encoder_hidden_states=image_embeds,
                                                        encoder_attention_mask=image_atts) 
          
                                                       
            cap_embeds = decoder_output_cg.hidden_states[-1].repeat_interleave(num_beams,dim=0)   
            cap_atts = torch.ones(cap_embeds.size()[:-1],dtype=torch.long).to(image.device)  

            captr_embeds = torch.cat((cap_embeds, triplet_states), dim=1)
            captr_atts = torch.cat((cap_atts, triplet_atts), dim=1)

            model_kwargs = {"encoder_hidden_states": captr_embeds, "encoder_attention_mask": captr_atts}
            bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)

            outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                    max_length=30,
                                                    min_length=10,
                                                    num_beams=num_beams,
                                                    eos_token_id=self.tokenizer.sep_token_id,
                                                    pad_token_id=self.tokenizer.pad_token_id, 
                                                    **model_kwargs)

            captions, questions = [], []    
            for output_cg in outputs_cg:
                caption = self.tokenizer.decode(output_cg, skip_special_tokens=True)    
                captions.append(caption[len(self.prompt):])
            for output in outputs:
                question = self.tokenizer.decode(output, skip_special_tokens=True)    
                questions.append(question)
            return captions, questions
            
    
    
def blip_vqg_c2q_onlyqg(pretrained='', pre_cgmodel='', freeze_img=False, **kwargs):
    model = BLIP_VQG_C2Q_QG(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        if pre_cgmodel:
            cgmodel, msg_cg = load_checkpoint(model,pre_cgmodel)
            model.visual_encoder.load_state_dict(cgmodel.visual_encoder.state_dict())
            model.text_decoder_cg.load_state_dict(cgmodel.text_decoder.state_dict())
        if freeze_img:
            for param in model.visual_encoder.parameters():
                 param.requires_grad = False
            for param in model.text_decoder_cg.parameters():
                param.requires_grad = False
#         assert(len(msg.missing_keys)==0)
    return model  


        
        
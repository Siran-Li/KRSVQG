import numpy as np
import pandas as pd
import json
import spacy
import argparse
import yaml
import os
import conceptnet_lite
from conceptnet_lite import Label, edges_for
from keytotext import pipeline
from random import randint
from tqdm import tqdm
import srsly

import itertools
from sentence_transformers import SentenceTransformer, util
import enchant
import re
from transformers import pipeline


conceptnet_lite.connect("ConceptNet/conceptnet.db")

def sort_dict_by_value(d, reverse = False, k=10):
  """Sort the dictionary by the values"""
  
  sorted_pairs = dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))

  return dict(itertools.islice(sorted_pairs.items(), k))

def suggest_obj(sug_word):
    """"Select the noun word"""
    
    sug_tokens = nlp(sug_word)
    for tok in sug_tokens:
        if tok.pos_ == "NOUN":            
            return tok.lemma_
        
def jsonline_reader(filename: str):
    """Load the dataset"""
    with open(filename, 'r') as f_reader:
        examples = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]
    return examples

def extract_core(doc):
    """Extract core words and the connected description"""
    
    noun_adj_pairs = []
    num_noun = 0
    regex = re.compile(r"'\s+(.*?)\s+'")
    
    for chunk in doc.noun_chunks:
        if chunk.text not in doc.text:
          continue
        adj = []
        noun = ''
        noun_words = ''
        continue_noun = 0
        for tok in chunk:
            if tok.pos_ == "NOUN":
                noun = tok.text.lower()
                obj = tok.lemma_
                if noun_words == '':
                    noun_words = obj
                    continue_noun += 1
                elif continue_noun>0:
                    noun_words += ' ' + obj 
            else:
                noun_words = ''
                continue_noun = 0

            if adj:
                adj.append(tok.text.lower())
            elif tok.pos_ == "ADJ" or tok.pos_ == "NUM" or tok.pos_ == "NOUN" or tok.pos_ == "CCONJ" or tok.pos_ == 'VERB' or tok.pos_ == 'PROPN':
                adj.append(tok.text.lower())

        if noun:
            des = " ".join(adj) 
            des_list = des.split('-')
            des_list = [des_i.strip() for des_i in des_list]
            des = '-'.join(des_list)
            des = regex.sub(r"'\1'", des)
            noun_adj_pairs.append({'id': num_noun, 'obj': obj, 'des': des})
            num_noun += 1
            if continue_noun>1:
                noun_adj_pairs.append({'id': num_noun, 'obj': noun_words, 'des': des})
                num_noun += 1
                
    return noun_adj_pairs

def replace_relation(knowledge):
    """Connect the knowledge triplet to a simple sentence"""
    
    if knowledge[1] == 'used for':
        tmp = 'is used for'
    elif knowledge[1] == 'receives action':
        tmp = 'receives action'
    elif knowledge[1] == 'has a':
        tmp = 'has a'
    elif knowledge[1] == 'causes':
        tmp = 'causes'
    elif knowledge[1] == 'has property':
        tmp = 'has a property'
    elif knowledge[1] == 'created by':
        tmp = 'is created by'
    elif knowledge[1] == 'defined as':
        tmp = 'is defined as'
    elif knowledge[1] == 'at location':
        tmp = 'is at location of'
    elif knowledge[1] == 'has subevent':
        tmp = 'has'
    elif knowledge[1] == 'made of':
        tmp = 'is made of'
    elif knowledge[1] == 'has prerequisite':
        tmp = 'has prerequisite to'
    elif knowledge[1] == 'desires':
        tmp = 'desires'
    elif knowledge[1] == 'is a':
        tmp = 'is a'
    elif knowledge[1] == 'not desires':
        tmp = 'not desires'
    elif knowledge[1] == 'capable of':
        tmp = 'is capable of'
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise Exception(knowledge[1])
    return knowledge[0] + ' ' + tmp + ' ' + knowledge[2]

def save_triplets(data, triplet_path):
    """Generate triplets and save to the csv file"""

    last_img = ''
    init_cap = 0
    num_raw = 0
    for item_i in tqdm(range(len(data))):
        image = data[item_i]['img_fname']
        
        # each image corresponds to 5 captions
        if last_img != image:
            if num_raw != 4 and num_raw != 0:
                raise Exception('Wrong image-caption matching')
            new_img = True
            last_img = image
            num_raw = 0
        else:
            new_img = False
            num_raw += 1

        sentence = data[item_i]['caption']

        if new_img:            
            top_triplets = []
            top_objects = []
            top_split = []
            top_predicts = []
            top_sim = []
            top_caps = []
            top_des = []

        # Find the nouns and verbs (keywords) with their description in each caption
        sentence = sentence.lower().replace(',', '')
        doc = nlp(sentence)
        pairs = extract_core(doc)
        descriptions = [pairs[p_idx]['des'] for p_idx in range(len(pairs))]
        df_des = pd.DataFrame(descriptions, columns=['des'])
        df_des['obj'] = [pairs[p_idx]['obj'] for p_idx in range(len(pairs))]
        df_des = df_des.drop_duplicates(keep='first').reset_index(drop=True)  
        
        embedding_1= senmodel.encode(sentence, convert_to_tensor=True)
        objects = list(df_des['obj'])  # the object word
        des_list = list(df_des['des']) # the object word with its description

        # get the relevant knowledge triplets from ConceptNet
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            des_i = des_list[obj_idx]
            # print('=====', obj, '=====')
            triplet_list = []
            pred_list = []
            tr_list = []
            try:
                edges = edges_for(Label.get(text=obj, language='en').concepts, same_language=True)
            except Label.DoesNotExist:
                sug_list = dict.suggest(obj)
                for sug_idx in range(len(sug_list)):
                    sug_words = sug_list[sug_idx]
                    obj = suggest_obj(sug_words)
                    # print('!-----------', obj, '-----------!')
                    # print('sub_word', obj)
                    try: 
                        edges = edges_for(Label.get(text=obj, language='en').concepts, same_language=True)
                        # exchange obj to the suggested one
                        df_des.at[obj_idx, 'obj'] = obj
                        break
                    except Label.DoesNotExist:
                        continue

            for e in edges:
                if e.relation.name in relation_list and e.etc['weight'] >= 1:
                    triplet = e.start.text + ' ' + e.relation.name + ' ' + e.end.text
                    triplet_list.append(triplet)
                    if e.start.text == obj:
                        pred_list.append(e.end.text.replace('_', ' '))
                        tr_list.append([des_i, e.relation.name.replace('_', ' '), e.end.text.replace('_', ' ')])
                    else:
                        pred_list.append(e.start.text.replace('_', ' '))
                        tr_list.append([e.start.text.replace('_', ' '), e.relation.name.replace('_', ' '), des_i])

            df_obj_triplet = pd.DataFrame(triplet_list, columns=['triplet'])
            df_obj_triplet['split'] = tr_list    # knowledge triplet
            df_obj_triplet['pred'] = pred_list   # the external entity
            df_obj_triplet = df_obj_triplet.drop_duplicates(subset=['triplet'], keep='first').reset_index(drop=True)
            df_obj_triplet['triplet'] = [l.replace('_', ' ') for l in list(df_obj_triplet['triplet'])]  

            triplet_list = list(df_obj_triplet['triplet'])

            # select triplets according to similarity with caption counted by SentenceTransformer
            if len(triplet_list)>0:

                sim_list = []
                for s_idx in range(len(triplet_list)):
                    trip_end = senmodel.encode(triplet_list[s_idx], convert_to_tensor=True)
                    sim_list.append(util.pytorch_cos_sim(embedding_1, trip_end)[0][0].cpu().detach().numpy())

                df_obj_triplet['obj_sim'] = sim_list
                df_obj_triplet = df_obj_triplet[df_obj_triplet['obj_sim']>=0.2]
                df_obj_triplet = df_obj_triplet[df_obj_triplet['obj_sim']<0.8] # avoid duplicated info <0.5
                df_obj_triplet = df_obj_triplet.sort_values('obj_sim', ascending=False).reset_index(drop=True)

                        
                if len(df_obj_triplet) > 10:
                    num_obj = 10
                else:
                    num_obj = len(df_obj_triplet)                                

                top_triplets += list(df_obj_triplet.loc[:(num_obj-1), 'triplet'])
                top_objects += [obj for oi in range(num_obj)]
                top_split += [df_obj_triplet.loc[oi, 'split'] for oi in range(num_obj)]
                top_predicts += [df_obj_triplet.loc[oi, 'pred'] for oi in range(num_obj)]
                top_sim += [df_obj_triplet.loc[oi, 'obj_sim'] for oi in range(num_obj)]
                top_caps += [sentence.lower() for oi in range(num_obj)]
                top_des += [df_des[df_des['obj']==obj]['des'].values[0] for oi in range(num_obj)]
                
        if num_raw == 4:
            # select the top triplets by the topic similarity between the caption and the external entities
            df_triplets = pd.DataFrame(top_triplets, columns=['triplet'])
            df_triplets.insert(loc=0, column='caption', value=top_caps)
            df_triplets['obj'] = top_objects
            df_triplets['des'] = top_des
            df_triplets['split'] = top_split
            df_triplets['pred'] = top_predicts
            df_triplets['obj_sim'] = top_sim


            if len(top_triplets) > 0:
                for t_idx in range(len(top_triplets)):
                    df_triplets.at[t_idx,'pred_score'] = classifier(top_caps[t_idx], candidate_labels=[top_predicts[t_idx]],)['scores'][0]

                df_triplets= df_triplets.drop_duplicates(subset=['triplet'], keep='first') 
                df_triplets = df_triplets[df_triplets['pred_score'] >= 0.2]  
                df_triplets = df_triplets[df_triplets['pred_score'] < 0.8]
                df_triplets = df_triplets.sort_values('obj_sim', ascending=False).reset_index(drop=True)


                df_questions = df_triplets.copy()
                if len(df_questions) > 10:
                    df_questions = df_questions.loc[:9, :]
                    num_q = 10
                else:
                    num_q = len(df_questions)

                num_q = len(df_questions)
                if num_q == 0:
                    txt.writelines(image, ' Non-question: '+ obj + ' -> ' + sentence + '\n')

                if num_q > 0:
                    # keep the top 10 triplets and make a simple knowledge sentence for each triplet
                    df_questions = df_questions.iloc[:num_q, :]
                    k2t_list = []
                    for num_t in range(num_q):
                        text = replace_relation(df_questions.loc[num_t, 'split'])
                        
                        k2t_list.append(text)

                    df_questions['k2t'] = k2t_list

                    answers = [df_questions.loc[tr_idx, 'des'].strip() for tr_idx in range(num_q)]
                    answers = [" ".join(answer.lower().split()) for answer in answers]

                    img_list = [image for tr_idx in range(len(df_questions))]
                    df_questions.insert(loc=0, column='img_fname', value=img_list)


                    if init_cap == 0:
                        df_info = df_questions
                    else:
                        df_info = pd.concat([df_info, df_questions])
                    init_cap += 1
                        

                    df_info.to_csv(triplet_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--device', default='cuda')  
   
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    device = args.device

    nlp = spacy.load('en_core_web_sm')
    dict = enchant.Dict('en_US')
    relation_list = ['has_a', 'used_for', 'capable_of', 'at_location', 'has_subevent', 
                     'has_prerequisite', 'has_property', 'causes', 'created_by', 'defined_as', 
                     'desires', 'made_of', 'not_desires', 'receives_action'] 
    
    senmodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')   
    classifier = pipeline(model="facebook/bart-large-mnli")

    file_path = config['nwpu_root']
    img_path = config['nwpu_img']
    triplet_path = config['nwpu_triplets']
    
    os.makedirs(triplet_path, exist_ok=True)
    os.makedirs(config['save_recording'], exist_ok=True)
    txt_path = os.path.join(config['save_recording'], 'nwpu_recording.txt')
    txt = open(txt_path,"w+")
    
    files = {'train': 'nwpu_train_ctq.json.gz', 'val': 'nwpu_val_ctq.json.gz'}
    for p, file in files.items():
        txt.writelines(p + '\n')
        data = srsly.read_gzip_json(f"{file_path}/{file}")
        trip_path = os.path.join(triplet_path, f'df_{p}_triplets.csv')

        save_triplets(data, trip_path)

#!/usr/bin/env python
# coding: utf-8

# from PIL import Image
import itertools, random
import glob,os, json
import random,shutil
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from balance_training_set import read_data,get_data,split_train_data

# ### Identifying gender nouns, their associated verbs 

# ### Define gender nouns, pronouns (expand/reduce lists as required)

import nltk
from nltk import word_tokenize, pos_tag
from collections import defaultdict

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

gender_nouns = ['boy', 'brother', 'dad', 'husband', 'man', \
        'groom', 'male','guy', 'men', 'males', 'boys', 'guys', 'dads', 'dude', \
        'policeman', 'policemen', 'father', 'son', 'fireman', 'actor', \
        'gentleman', 'boyfriend', 'mans', 'his', 'obama', 'actors'
    'girl', 'sister', 'mom', 'wife', 'woman', \
        'bride', 'female', 'lady', 'women', 'girls', 'ladies', 'females', \
        'moms', 'actress', 'nun', 'girlfriend', 'her', 'she']               

gender_pronouns = []
verb_tags = ['VBG','VBD','VBN','VB']
adj_tags = ['JJ','JJR','JJS']
noun_tags = ['NN','NNS','NNP','NNPS']

def get_bias_files(act):
    path = ["./training bias/",act+".txt"]
    files = glob.glob("*".join(path))
    return files

def get_bias_dict(activity):
    files = get_bias_files(activity)
    bias_dicts = {}
    for i,file in enumerate(files):
        with open(file) as f:
            print(file)
            i = 'woman' if 'female' in file else 'man'
            bias_dicts[i] ={line.split()[0]:float(line.split()[1]) for line in f}
    return bias_dicts['man'],bias_dicts['woman']

def get_traincaptions(ids, n_samples):
    
    target_folder = 'trainfolder/'
    train_set = []
    filenames = {}
    captions = {}
    print("Getting samples: ",n_samples, "nos.")
    
    train_set = [id for id in random.sample(ids, n_samples)]
#     print(train_set[:5])
    with open('captions_train2014.json') as file:
        coco_data = json.load(file)
    for item in coco_data['images']:
        if len(filenames) == n_samples:
            
            break
        if item['id'] in train_set:
            
            filenames[item['id']] = item['file_name']
    
    for item in coco_data['annotations']:
        if len(captions)==n_samples:
            
            break
        if item['image_id'] in train_set:
            
            filename = filenames[item['image_id']]
            if filename in captions:
                captions[filename] += [item['caption']]
            else:
                captions[filename] = [item['caption']]
    print(len(captions))
    # for image in captions.keys():
    #     image = "train2014/" + image 
    #     shutil.copy(image, target_folder)
    # print("Copied sampled images to {}".format(target_folder))
    return captions,train_set
    
def get_tag_dicts(get_tags,all_captions):
    gender_tags_dict = defaultdict(list)
    pronoun_tags_dict = defaultdict(list)
#     print(pronoun_tags_dict)
    # gender_tags_dict = dict.fromkeys(gender_nouns, [])
    # pronoun_tags_dict = dict.fromkeys(gender_pronouns, [])

    for caption in all_captions:
        # verbs : VBG, VBD, VBN
        tokens = word_tokenize(caption)

        if any(nn in tokens for nn in gender_nouns) or any(pn in tokens for pn in gender_pronouns):
            tags = pos_tag(tokens)

            only_tags = [tag[1] for tag in tags]
            if any(get_tag in only_tags for get_tag in get_tags):
                only_verbs = [tag[0] for tag in tags if tag[1] in get_tags]
                if set(gender_nouns).intersection(tokens):   

                    for nn in set(gender_nouns).intersection(tokens):

                        gender_tags_dict[nn].extend(only_verbs)

                if set(gender_pronouns).intersection(tokens):
                    for pn in set(gender_pronouns).intersection(tokens):
    #                     print(pn, pronoun_tags_dict[pn])
                        pronoun_tags_dict[pn].extend(only_verbs)
    
    return gender_tags_dict, pronoun_tags_dict


# ### lists of words associated with gender nouns and pronouns

# ### Most frequent items (verbs/adj/nouns) associated with each gender noun/pronoun


def top_n_items(nn='woman',pos='verb',count=0,all_captions=[]):
    global gender_count, pronoun_count
    corresp_tags = {'adj': adj_tags, 'verb':verb_tags, 'noun':noun_tags}
    gender_dict, pronoun_dict = get_tag_dicts(corresp_tags[pos],all_captions)  
    gender_count,pronoun_count = {},{}
    
    for (key1, value1) in gender_dict.items():
        gender_count[key1] = set([(i,value1.count(i)) for i in value1 if i not in gender_nouns])
    for (key2, value2) in pronoun_dict.items():   
        pronoun_count[key2] = set([(i,value2.count(i)) for i in value2 if i not in gender_nouns])     
    
    dictionary = gender_count if nn in gender_nouns else pronoun_count
    if count==0:
        return
    print(sorted(dict(dictionary[nn]), key = dict(dictionary[nn]).get, reverse=True)[:count])


def counts():
    global all_count,pro_all_count
    all_count = {}
    pro_all_count = {}    
    for item1 in gender_count:
        
        all_count[item1] = {each[0]:each[1] for each in gender_count[item1]}
    for item2 in pronoun_count:
        pro_all_count[item2]= {each[0]:each[1] for each in pronoun_count[item2]}

    all_count = split_by_gender(all_count)

def split_by_gender(all_count):

    binary_dict = {'man':{},'woman':{}}
    male_nouns = ['boy', 'brother', 'dad', 'husband', 'man', \
        'groom', 'male','guy', 'men', 'males', 'boys', 'guys', 'dads', 'dude', \
        'policeman', 'policemen', 'father', 'son', 'fireman', 'actor', \
        'gentleman', 'boyfriend', 'mans', 'his', 'obama', 'actors']
    female_nouns = ['girl', 'sister', 'mom', 'wife', 'woman', \
        'bride', 'female', 'lady', 'women', 'girls', 'ladies', 'females', \
        'moms', 'actress', 'nun', 'girlfriend', 'her', 'she'] 
    for key,value in all_count.items():
        if key in male_nouns:
            nn = 'man'
        elif key in female_nouns:
            nn = 'woman'
        else:
            continue
        for key2, value2 in value.items():
            try:
                binary_dict[nn][key2] += value2
            except: binary_dict[nn][key2] = value2

    return binary_dict



def bias(gender1 = 'man',gender2 = 'woman'):
    global bias_dict1,bias_dict2,data
    # data : gender nouns or pronouns : all_count or pro_all_count
#     data = all_count.copy() if gender in gender_nouns else pro_all_count.copy()
    counts()
    
    data = all_count.copy()
    
    dictionary = set([a for key,values in data.items() for a in values if key==gender1 or key==gender2])    
     
    bias_dict1, bias_dict2 = {}, {} 
    for word in dictionary:
        try:gender1_count = data[gender1].get(word,0)
        except:gender1_count=0
        try:gender2_count = data[gender2].get(word,0)
        except:gender2_count=0
        if gender1_count!=0 : bias_dict1[word] = gender1_count/(gender1_count+gender2_count)
        if gender2_count!=0: bias_dict2[word] = gender2_count/(gender1_count+gender2_count)            

    return bias_dict1, bias_dict2





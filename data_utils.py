import glob
import pandas as pd
import re
import json
import nltk
import numpy as np
import random
from utils import agreement_score, export_csv, save_obj, load_obj
from sklearn.model_selection import StratifiedShuffleSplit

def get_activity_list(save_file = False):
    # Tagged lists of Activity/ object images from kayburns' Github Repo "Women Snowboard"
    # https://github.com/kayburns/women-snowboard
    activity_list_paths = glob.glob("./data/list/intersection_*")
    activity_image_ids = dict()
    for path in activity_list_paths:
        with open(path, 'r') as f:
            p = re.compile(r'(?<=_)(.*)(?=_)')
            activity = p.findall(path)[0]
            im_ids= f.read().split('\n')
            im_ids = [int(i) for i in im_ids if i != '']
            activity_image_ids[activity] = im_ids
    
    if save_file == True:
        save_obj(activity_image_ids, 'activity_image_ids')
    else:
        return activity_image_ids

def get_gender_nouns(save_file = False):
    # Gender word lists from sueqian6's Github Repo "Reducing Gender Bias in Word-level Language Models"
    # https://github.com/sueqian6/ACL2019-Reducing-Gender-Bias-in-Word-Level-Language-Models-Using-A-Gender-Equalizing-Loss-Function

    gender_list_paths = glob.glob("./data/list/*_word_file.txt")
    gender_list_paths.append('./data/list/neutral_occupations.txt')
    gender_nouns_lookup = dict()
    for path in gender_list_paths:
        with open(path, 'r') as f:
            if path == './data/list/neutral_occupations.txt':
                gender = 'neutral'
            else:
                p = re.compile(r'(?<=list/)(.*)(?=_word)')
                gender = p.findall(path)[0]
            nouns = f.read().split('\n')
            nouns = [n for n in nouns if n != '']
            gender_nouns_lookup[gender] = nouns

    # There are some inappropriate in the word lists, we are doing some hand crafting to avoid adding bias.
    # Remove non-human words
    for word in ['cow', 'cows', 'hen', 'hens']:
        gender_nouns_lookup['female'].remove(word)
    for word in ['bull', 'bulls', 'lion', 'lions', 'governor']:
        gender_nouns_lookup['male'].remove(word) 
    # Add gender-neutral words
    for word in ['surfer', 'child', 'kid', 'kids', 'children', 'passenger', 'passengers',\
        'governor', 'someone', 'pedestrian', 'pedestrians']:
        gender_nouns_lookup['neutral'].append(word)

    if save_file == True:
        save_obj(gender_nouns_lookup, 'gender_nouns_lookup')
    else:
        return gender_nouns_lookup

def get_qualified_dataset(annotations_path, save_file = False):
    '''
    captions_dict (dict)- key: image_id, value: list of captions

    im_gender_summary (dict of dict)- key: image_id, value: dict()
    keys in dict: pred_gt- predicted ground truth label of the gender noun
                per_gt- % of annotations (out of 5 total) that agreed with the GT
                agreement_score- agreement score calculated using distance between 5 predictions, with 1 being the best
                                male = 1, female = -1, neutral = 0
                                e.g.0 annotations indicate [f, f, f, f, f], agreement_score = 1.00
                                e.g.1 annotations indicate [m, m, f, f, f], agreement_score = 0.00
                                e.g.2 annotation indicate [n, n, f, f, f], agreement_score = 0.50                                                        
                anno_gender- list of gender sentiment, e.g. ['male', 'female', 'neutral', 'female', 'female']
                anno_nouns- list of nouns used to describe human
                clean_gender- binary variable indicating if all notations used the same gender/ gender-neutral noun 
                clean_noun- binary variable indicating if all notations used the identical noun

    not_human_im_ids(list)- list of image ids of images with >1 captions that do not mention humans.
    Since the COCO dataset does not label whether human (or other objects) is the major subject 
    matter of the image. This list helps us isolate images with human figures as the focus.
    '''
    captions_dict = dict()
    im_gender_summary = dict()
    not_human_im_ids = list() 

    # load pre-processed data
    gender_nouns_lookup = load_obj('gender_nouns_lookup')

    for datatype in ['train', 'val']:
        print(f"\nEvaluating ground truth labels in {datatype} set")
        with open(f'{annotations_path}/captions_{datatype}2014.json') as f:
            captions_json = json.load(f)

            for i in range(len(captions_json['annotations'])):
                image_id = captions_json['annotations'][i]['image_id']
                caption = captions_json['annotations'][i]['caption']
                tokens = nltk.word_tokenize(caption)
                c_female = 0 # count of gender nouns and gender-neutral nouns
                c_male = 0
                c_neutral = 0
                noun = []

                # Evaluate annotator's noun used to describe humans
                for t in tokens:
                    t = t.lower()
                    if t in gender_nouns_lookup['female']:
                        c_female += 1
                        noun.append(t)
                    elif t in gender_nouns_lookup['male']:
                        c_male += 1
                        noun.append(t)
                    elif t in gender_nouns_lookup['neutral']:
                        c_neutral += 1
                        noun.append(t)

                # Only include image for training if more than one caption of the image mention human
                # Conflicting gender mentions are also dropped, e.g. "a boy and a girl are on a beach"
                if c_female + c_male + c_neutral == 1:
                    # Assign gender sentiment to the caption
                    if c_female > 0:
                        gender = 'female'
                    elif c_male > 0:
                        gender = 'male'
                    else:
                        gender = 'neutral'

                    # Populate captions dict and image gender summary dict
                    if image_id in captions_dict:
                        captions_dict[image_id] += [caption]
                        im_gender_summary[image_id]['anno_gender'].append(gender)
                        im_gender_summary[image_id]['anno_noun'].append(noun[0])
                    else:
                        captions_dict[image_id] = [caption]
                        im_gender_summary[image_id] = dict()
                        im_gender_summary[image_id]['anno_gender'] = [gender]
                        im_gender_summary[image_id]['anno_noun'] = [noun[0]]

                if i % 100000 == 0:
                    print()
                    print(f"Caption {i} processed, out of {len(captions_json['annotations'])} captions")
                    print(f"No. of qualified images processed: {len(im_gender_summary)}")

    for image_id in im_gender_summary:
        # Delete images where <3 annotators mentioned the human figure
        # Because it is impossible to estimate the ground truth using only 1 or 2 captions 
        if len(im_gender_summary[image_id]['anno_gender']) < 3:
            not_human_im_ids.append(image_id)
        
        else:
            pred = im_gender_summary[image_id]['anno_gender']

            # Evaluate groundtruth guesses and agreement scores
            gt = max(set(pred), key = pred.count)

            # Populate dictionary
            im_gender_summary[image_id]['pred_gt'] = gt
            im_gender_summary[image_id]['per_gt'] = sum([1 for p in pred if p == gt]) / len(pred)
            im_gender_summary[image_id]['agreement_score'] = agreement_score(pred)
            if len(set(pred)) == 1:
                im_gender_summary[image_id]['clean_gender'] = 1
            else:
                im_gender_summary[image_id]['clean_gender'] = 0
            if len(set(im_gender_summary[image_id]['anno_noun'])) == 1:
                im_gender_summary[image_id]['clean_noun'] = 1
            else:
                im_gender_summary[image_id]['clean_noun'] = 0
            
    for image_id in not_human_im_ids:
        try:
            del captions_dict[image_id]
            del im_gender_summary[image_id]
        except:
            pass
    
    if save_file == True:
        export_csv('./data/list/qualified_image_ids.csv', list(im_gender_summary.keys()))
        save_obj(captions_dict, 'captions_dict')
        save_obj(im_gender_summary, 'im_gender_summary')
    else:
        return captions_dict, im_gender_summary

def get_training_indices(sample_size, mode = 'random'):
    assert mode in ['random','balanced_mode','balanced_clean', 'balanced_gender_only', \
                    'balanced_clean_noun', 'clean_noun', 'activity_balanced', 'activity_balanced_clean']
    assert isinstance(sample_size, int)
    '''
    8 different modes of generating data
    - random: randomized selection of qualified images
    - balanced_mode: balanced ratio between male, female and neutral
    - balanced_clean: balanced ratio between male, female and neutral,
                      only use images when all captions agree on using the same gender
    - balanced_gender_only: same as balanced_mode, but without neutral captions
    - balanced_clean_noun: balanced ratio between male, female and neutral, only use images when all captions
                           agree on using the same noun
    - clean_noun: only use images when all captions agree on the same noun
    - activity_balanced: from activity tagged image sets, choose same ratio of male, female, neutral image
    - activity_balanced_clean: similar to activity_balanced, but all captions must agree on the same gender
    
    Note that it is possible that output size may be smaller than sample_size,
    especially for activity_balanced and activity_balanced_clean. As for certain activities, the sample size of
    clean data might be limited for some classes, e.g. women wearing tie.
    '''
    
    random.seed(123)
    training_captions_dict = dict()

    # Get pre-processed objects
    im_gender_summary = load_obj('im_gender_summary')
    captions_dict = load_obj('captions_dict')
    activity_image_ids = load_obj('activity_image_ids')
    
    if mode == 'random':
        training_captions_dict = dict(random.sample(captions_dict.items(), sample_size))
        
    elif mode == 'balanced_mode':
        i = 0
        male_count = 0
        female_count = 0
        neutral_count = 0
        for image_id in im_gender_summary.keys():
            if i < sample_size:
                if im_gender_summary[image_id]['pred_gt'] == 'male' and (male_count < sample_size / 3):
                    training_captions_dict[image_id] = captions_dict[image_id]
                    male_count += 1
                    i += 1
                elif im_gender_summary[image_id]['pred_gt'] == 'female' and (female_count < sample_size / 3):
                    training_captions_dict[image_id] = captions_dict[image_id]
                    female_count += 1
                    i += 1
                elif im_gender_summary[image_id]['pred_gt'] == 'neutral'and (neutral_count < sample_size / 3):
                    training_captions_dict[image_id] = captions_dict[image_id]
                    neutral_count += 1
                    i += 1
                    
                if i % 1000 == 0:
                    print(f"captions of {i} images are added")
    
    elif mode == 'balanced_clean':
        i = 0
        male_count = 0
        female_count = 0
        neutral_count = 0
        for image_id in im_gender_summary.keys():
            if i < sample_size:
                if im_gender_summary[image_id]['clean_gender'] == 1:
                    if im_gender_summary[image_id]['pred_gt'] == 'male' and (male_count < sample_size / 3):
                        training_captions_dict[image_id] = captions_dict[image_id]
                        male_count += 1
                        i += 1
                    elif im_gender_summary[image_id]['pred_gt'] == 'female' and (female_count < sample_size / 3):
                        training_captions_dict[image_id] = captions_dict[image_id]
                        female_count += 1
                        i += 1
                    elif im_gender_summary[image_id]['pred_gt'] == 'neutral'and (neutral_count < sample_size / 3):
                        training_captions_dict[image_id] = captions_dict[image_id]
                        neutral_count += 1
                        i += 1
                    
                if i % 1000 == 0:
                    print(f"captions of {i} images are added")
    
    elif mode == 'balanced_clean_noun':
        i = 0
        male_count = 0
        female_count = 0
        neutral_count = 0
        for image_id in im_gender_summary.keys():
            if i < sample_size:
                if im_gender_summary[image_id]['clean_noun'] == 1:
                    if im_gender_summary[image_id]['pred_gt'] == 'male' and (male_count < sample_size / 3):
                        training_captions_dict[image_id] = captions_dict[image_id]
                        male_count += 1
                        i += 1
                    elif im_gender_summary[image_id]['pred_gt'] == 'female' and (female_count < sample_size / 3):
                        training_captions_dict[image_id] = captions_dict[image_id]
                        female_count += 1
                        i += 1
                    elif im_gender_summary[image_id]['pred_gt'] == 'neutral'and (neutral_count < sample_size / 3):
                        training_captions_dict[image_id] = captions_dict[image_id]
                        neutral_count += 1
                        i += 1
                    
                if i % 1000 == 0:
                    print(f"captions of {i} images are added") 
                    
    elif mode == 'clean_noun':
        i = 0
        for image_id in im_gender_summary.keys():
            if i < sample_size:
                if im_gender_summary[image_id]['clean_noun'] == 1:
                    training_captions_dict[image_id] = captions_dict[image_id]
                    i += 1
                    
                if i % 1000 == 0:
                    print(f"captions of {i} images are added")   
    
    elif mode == 'balanced_gender_only':
        i = 0
        male_count = 0
        female_count = 0
        for image_id in im_gender_summary.keys():
            if i < sample_size:
                if im_gender_summary[image_id]['pred_gt'] == 'male' and (male_count < sample_size / 2):
                    training_captions_dict[image_id] = captions_dict[image_id]
                    male_count += 1
                    i += 1
                elif im_gender_summary[image_id]['pred_gt'] == 'female' and (female_count < sample_size / 2):
                    training_captions_dict[image_id] = captions_dict[image_id]
                    female_count += 1
                    i += 1
                    
                if i % 1000 == 0:
                    print(f"captions of {i} images are added")
    
    elif mode == 'activity_balanced':
        activity_sample_size = sample_size / len(activity_image_ids.keys())
        i = 0
        for activity in activity_image_ids.keys():
            image_ids = activity_image_ids[activity]
            j = 0
            male_count = 0
            female_count = 0
            neutral_count = 0
            for image_id in image_ids:
                if j < activity_sample_size:
                    if image_id in im_gender_summary:
                        if im_gender_summary[image_id]['pred_gt'] == 'male' and (male_count < activity_sample_size / 3):
                            training_captions_dict[image_id] = captions_dict[image_id]
                            male_count += 1
                            i += 1
                            j += 1
                        elif im_gender_summary[image_id]['pred_gt'] == 'female' and (female_count < activity_sample_size / 3):
                            training_captions_dict[image_id] = captions_dict[image_id]
                            female_count += 1
                            i += 1
                            j += 1
                        elif im_gender_summary[image_id]['pred_gt'] == 'neutral'and (neutral_count < activity_sample_size / 3):
                            training_captions_dict[image_id] = captions_dict[image_id]
                            neutral_count += 1
                            i += 1
                            j += 1

                    if i > 0 and i % 100 == 0:
                        print(f"captions of {i} images are added")
    
    elif mode == 'activity_balanced_clean':
        activity_sample_size = sample_size / len(activity_image_ids.keys())
        i = 0
        for activity in activity_image_ids.keys():
            image_ids = activity_image_ids[activity]
            j = 0
            male_count = 0
            female_count = 0
            neutral_count = 0
            for image_id in image_ids:
                if j < activity_sample_size:
                    if image_id in im_gender_summary and im_gender_summary[image_id]['clean_noun'] == 1:
                        if im_gender_summary[image_id]['pred_gt'] == 'male' and (male_count < activity_sample_size / 3):
                            training_captions_dict[image_id] = captions_dict[image_id]
                            male_count += 1
                            i += 1
                            j += 1
                        elif im_gender_summary[image_id]['pred_gt'] == 'female' and (female_count < activity_sample_size / 3):
                            training_captions_dict[image_id] = captions_dict[image_id]
                            female_count += 1
                            i += 1
                            j += 1
                        elif im_gender_summary[image_id]['pred_gt'] == 'neutral'and (neutral_count < activity_sample_size / 3):
                            training_captions_dict[image_id] = captions_dict[image_id]
                            neutral_count += 1
                            i += 1
                            j += 1

                        if i > 0 and i % 1000 == 0:
                            print(f"captions of {i} images are added")
    
    training_image_ids = list(training_captions_dict.keys())
    return training_image_ids, training_captions_dict

def train_test_split(training_image_ids, test_size = 0.3, random_state = 123):

    # Get pre-processed objects
    im_gender_summary = load_obj('im_gender_summary')

    X = np.asarray(training_image_ids)
    y = np.asarray([im_gender_summary[x]['pred_gt'] for x in X])
    # Use Stratified shuffle split to ensure the ratio of gender ratio stays the same in train set and validation set (both balanced or random)
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_size, random_state = random_state)
    for train_idx, test_idx in sss.split(X, y):
        train_image_ids, test_image_ids = X[train_idx], X[test_idx]
        gender_train, gender_test = y[train_idx], y[test_idx]
    return train_image_ids, test_image_ids, gender_train, gender_test
    # Output X: list of image ids, Y: gender class of test and train images
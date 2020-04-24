import json
import time
import numpy as np
import copy
import itertools
import os
from collections import defaultdict
import sys
from pprint import pprint
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import pickle
from model_utils import predict_from_COCO
from utils import load_obj

def confusion_matrix(test_pred_captions):

    ##Get the captions dictionary
    captions_dict = load_obj('captions_dict')
    ##Get the gender summary dictionary 
    im_gender_summary = load_obj('im_gender_summary')
    ##Get the nouns associated with each gender
    gender_nouns_lookup = load_obj('gender_nouns_lookup')
    male_tags=gender_nouns_lookup['male']
    female_tags=gender_nouns_lookup['female']
    neutral_tags=gender_nouns_lookup['neutral']

    ##Initialize the confusion matrix
    conf_matrix=[[0]*3]*3

    for test_image_id in list(test_pred_captions.keys()):
        ##Get the ground truth tag for the test image
        gt = im_gender_summary[test_image_id]['pred_gt']

        ##Get the predicted gender tag for the test image
        #get predicted captions first
        img_capts = test_pred_captions[test_image_id]

        ##Construct conf matrix
        if sum(any(word in nltk.word_tokenize(caption) for word in male_tags)\
            for caption in img_capts)>1:
            pred='male'
        elif sum(any(word in nltk.word_tokenize(caption) for word in female_tags)\
            for caption in img_capts)>1:
            pred='female'
        else:
            pred='neutral'

        if gt=='male':
            if pred==gt:
                conf_matrix[0][0]+=1
            if pred=='female':
                conf_matrix[0][1]+=1
            if pred=='neutral':
                conf_matrix[0][2]+=1
        if gt=='female':
            if pred=='male':
                conf_matrix[1][0]+=1
            if pred==gt:
                conf_matrix[1][1]+=1
            if pred=='neutral':
                conf_matrix[1][2]+=1
        if gt=='neutral':
            if pred=='male':
                conf_matrix[2][0]+=1
            if pred=='female':
                conf_matrix[2][1]+=1
            if pred==gt:
                conf_matrix[2][2]+=1
                
    return conf_matrix
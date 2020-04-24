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
from src.model_utils import predict_from_COCO, predict_for_test_samples
from src.utils import load_obj, caption_to_gender
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

def eval_model(image_folder_path, vocab_path = '', model_path = '', training_image_ids_path = '', sample_size = 100, mode = 'balanced_mode'):
    test_pred_captions = predict_for_test_samples(image_folder_path = image_folder_path,\
                         sample_size = sample_size, vocab_path = vocab_path, model_path = model_path,\
                           training_image_ids_path = training_image_ids_path, mode = 'balanced_mode')

    # load objects
    im_gender_summary = load_obj("im_gender_summary")
    captions_dict = load_obj("captions_dict")
    
    # populate ref_captions
    ref_captions = []
    for _, captions in captions_dict.items():
        for c in captions:
            ref_captions.append(c.split())
    # intiaite list
    gt = []
    pred_gender = []
    bleus = []
    
    for image_id, caption in test_pred_captions.items():
        gt.append(im_gender_summary[image_id]['pred_gt'])
        pred_gender.append(caption_to_gender(caption))
        bleus.append(sentence_bleu(ref_captions, caption.split()))
        
    labels = ['male', 'neutral', 'female']
    conf_matrix = confusion_matrix(gt, pred_gender, labels = labels)
    confusion_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    accuracy = accuracy_score(gt, pred_gender)
    bleu = np.mean(bleus)

    print(f"Confusion Matrix (columns = Ground Truth gender, rows = Predicted gender):\n{confusion_matrix_df}")
    print(f"\nAccuracy score: {accuracy}")
    print(f"\nTest Bleu Score (compared against original human labels of test set):{bleu}")

    return confusion_matrix_df, accuracy, bleu
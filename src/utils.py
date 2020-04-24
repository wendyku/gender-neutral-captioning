import numpy as np
import csv
import pickle
import nltk

'''
Agreement score calculated using distance between 5 predictions, with 1 being the best
male = 1, female = -1, neutral = 0
e.g.0 annotations indicate [f, f, f, f, f], agreement_score = 1.00
e.g.1 annotations indicate [m, m, f, f, f], agreement_score = 0.00
e.g.2 annotation indicate [n, n, f, f, f], agreement_score = 0.50

input: im_gender_summary[image_id]['anno_gender']
list of gender of noun labelled by the annotators
'''
def agreement_score(anno_gender):
    error = 0
    score_cal_dict ={
        'male':1, 'female':-1, 'neutral':0
    }

    # Calculate pairwise distance between each prediction and sum up errors
    for ind, p in enumerate(anno_gender):
        for other_p in [x for i,x in enumerate(anno_gender) if i != ind]:
            error += np.abs(score_cal_dict[p] - score_cal_dict[other_p])
    
    # Because there are only 3 classes available, but there can be 3-5 captions for each image.
    # max_error is diff based on number of captions
    if len(anno_gender) == 3:
        max_error = 8
    elif len(anno_gender) == 4:
        max_error = 16
    else:
        max_error = 24
    
    score = (max_error- error) / max_error
    # 24 is the max error because there are only 3 classes available, and there are 
    return score

def export_csv(filepath, data):
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    #print(f"saved in {filepath}")

def import_csv(filepath):
    with open('file.csv', newline='') as f:
        reader = csv.reader(f)
    #print(f"Loading {filepath}")
    return list(reader)

def save_obj(obj, name):
    with open('obj/'+ str(name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    #print(f"{name} saved  as ~/obj/{name}.pkl")

def load_obj(name):
    with open('obj/' + str(name) + '.pkl', 'rb') as f:
        #print(f"Loading {str(name)} from ~/obj/{str(name)}.pkl")
        return pickle.load(f)

def caption_to_gender(caption):
    # Load list
    gender_nouns_lookup = load_obj("gender_nouns_lookup")
    tokens = nltk.word_tokenize(caption)
    c_female = 0 # count of gender nouns and gender-neutral nouns
    c_male = 0
    c_neutral = 0

    # Evaluate annotator's noun used to describe humans
    for t in tokens:
        t = t.lower()
        if t in gender_nouns_lookup['female']:
            c_female += 1
        elif t in gender_nouns_lookup['male']:
            c_male += 1
        elif t in gender_nouns_lookup['neutral']:
            c_neutral += 1

    # Only include image for training if more than one caption of the image mention human
    # Conflicting gender mentions are also dropped, e.g. "a boy and a girl are on a beach"
    if c_female > 0:
        gender = 'female'
    elif c_male > 0:
        gender = 'male'
    else:
        gender = 'neutral'
    
    return gender

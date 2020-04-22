import numpy as np
import csv
import pickle

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
    score = (24 - error) / 24
    return score

def export_csv(filepath, data):
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    print(f"{str(data)} saved in {filepath}")

def import_csv(filepath):
    with open('file.csv', newline='') as f:
        reader = csv.reader(f)
    print(f"Loading {filepath}")
    return list(reader)

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(f"{name} saved  as ~/obj/{name}.pkl")

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        print(f"Loading {name} from ~/obj/{name}.pkl")
        return pickle.load(f)
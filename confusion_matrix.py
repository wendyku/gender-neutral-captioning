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

def predict_capts(test_image_ids, vocab_path = '', model_path = '',\
                 embed_size = 256, hidden_size = 512, mode = 'balanced_clean'):
    # Get model
    if model_path == '': # if not specified, assume it is best model saved in models
        model_path = './models/best-model.pkl'
    if torch.cuda.is_available() == True:
        checkpoint = torch.load('./models/best-model.pkl')
    else:
        checkpoint = torch.load('./models/best-model.pkl', map_location='cpu')
    
    # Get the vocabulary and its size
    if vocab_path == '': # if not specified, assume it is the vocab pickle saved in object
        vocab = load_obj('vocab')
    else:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    vocab_size = len(vocab)
  
    # convert image
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
    

    # Initialize the encoder and decoder, and set each to inference mode
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the pre-trained weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    predicted_capts=dict()

    test_loader = load_data(test_image_ids,\
                            image_folder_path, mode = 'test')
    

    for test_image_id in test_image_ids:
        ##Get the image path from the image_id
        original_image, image = next(iter(test_loader))
        
        transformed_image = transform(image)
        transformed_image_plot = np.squeeze(transformed_image.numpy())\
                    .transpose((1, 2, 0))
        features = encoder(transformed_image).unsqueeze(1)
        output = decoder.sample_beam_search(features)
        sentences = clean_sentence(output, vocab)
        ##add sentences to predicted image captions
        predicted_captions['test_image_id']=sentences

    return predicted_captions

##Define a function that takes the output image ids and generates
##the confusion matrix
def confusion_matrix(train_image_ids, test_image_ids, pred_mode= 'balanced_clean'):

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
    predicted_captions=predict_capts(test_image_ids, mode = pred_mode)

    for test_image_id in test_image_ids:
        ##Get the ground truth tag for the test image
        gt=im_gender_summary[test_image_id]['pred_gt']

        ##Get the predicted gender tag for the test image

        #get predicted captions first
        img_capts=predicted_capts['test_image_id']

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
            
if __name__=='__main__':
##    training_image_ids_path='./obj/training_image_ids.pkl'
##    with open(training_image_ids_path, 'rb') as f:
##        training_image_ids = pickle.load(f)
##        # print(training_image_ids)
##    
##    for i in range(5):
##        print(captions_dict[training_image_ids[i]])
##    print ([[0]*3]*3)
    pass

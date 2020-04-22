import torch
import nltk
import numpy as np
import pickle
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from data_utils import get_training_indices
from utils import load_obj
from Vocabulary import Vocabulary

class MyDataset(Dataset):
    
    def __init__(self, image_folder_path, mode = 'train', sample_size = 100,
              vocab_threshold = 5, batch_size = 10):
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        
        # Obtain sample of training images
        self.training_image_ids, captions_dict = get_training_indices(sample_size = sample_size, mode = "balanced_clean")
        
        # self.training_image_ids, self.images_path, self.image_id_dict, captions_dict \
        # = get_data(image_folder_path, annotations_path, sample_size, data_type)

        # Set up vocabulary or load from training set
        if self.mode == 'train':
            self.vocab = Vocabulary(captions_dict)
            print('Vocabulary successfully created')
        else:
            self.vocab = load_obj("vocab")
            self.word2idx = self.vocab.word2idx
            self.idx2word = self.vocab.idx2word
            print('Vocabulary successfully loaded')
        
        # Set up dataset
        self.im_ids = []
        self.captions = []
        self.images = []
        self.captions_len = []
        for im_id, captions_list in captions_dict.items():
            for item in captions_list:
                self.im_ids.append(im_id)
                self.captions.append(item)
                self.captions_len.append(len(nltk.tokenize.word_tokenize(item)))
        
        # Set up paramteres for image feature extraction 
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
    
    def __getitem__(self, index):
        im_id = self.im_ids[index] 
        l = len(str(im_id)) # for recreating the file name

        # Locate the image file in train or val
        try:
            image = Image.open(self.image_folder_path + "train2014/COCO_train2014_"+ "0"* (12-l) + str(im_id) + '.jpg').convert("RGB")
        except:
            try:
                image = Image.open(self.image_folder_path + "val2014/COCO_val2014_" + "0"* (12-l) + str(im_id) + '.jpg').convert("RGB")
            except:
                print(f"Image file {im_id} cannot be located")
                pass

        if self.mode == "train" or self.mode == 'val':
            # Convert image to tensor
            image = self.transform(image)
            
            # Tokenize captions
            tokens = nltk.tokenize.word_tokenize(str(self.captions[index]).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()
        
            return image, caption

        else: # if mode is test, return original and transformed image
            original_image = np.array(image)
            transformed_image = self.transform(image)
        
            return original_image, transformed_image

    
    def get_indices(self):
        search_len = np.random.choice(self.captions_len)
        all_indices = np.where([self.captions_len[i] == search_len\
                for i in range(len(self.captions_len))])[0]
        indices = list(np.random.choice(all_indices, size = self.batch_size))
        return indices
    
    def __len__(self):
        return len(self.ids)
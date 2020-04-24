# -*- coding: utf-8 -*-

## Evaluating mean bias amplification


# from balance_training_set import read_data,get_data,split_train_data
from biasanalysis.bias_analysis import top_n_items, bias


# top 12 biased verbs from both genders
bias_activities = ['driving','passing','catching','reading','touching','pulling','wedding','jumping','skateboarding','pulled','riding',
'pushing','cutting','having','laying','cut','petting','waiting','talking','haired','overlooking','staring','typing']

# top 12 biased nouns from both genders
bias_items = ['tennis','dirt','skateboard','grass','carriage','beer','kite','beach','road','wagon','bags','surfboard',
              'dress','lap','bed','flowers','mouth','bridle','fire','teeth','curb','device','leash','face']

def feature_bias(bias_dict, feature):
  return bias_dict.get(feature,0)

def is_biased(bias_dict,feature):
  return bias_dict.get(feature,0) > 0.5

def objects_bias(bias_dict1,bias_dict2,gender='man'):
 
  bias_dict = bias_dict1 if gender=='man' else bias_dict2
  objects_dict = {feature:feature_bias(bias_dict,feature) for feature in bias_items}
  return objects_dict

def bias_diff_sum(train_bias_dict, test_bias_dict):
  feature_diff = 0
  # train_bias_dict = train_dict_man if gender=='man' else train_dict_woman
  # test_bias_dict = test_dict_man if gender=='man' else test_dict_woman
  for feature in bias_items:
    feature_diff += feature_bias(test_bias_dict,feature) - feature_bias(train_bias_dict, feature)
  return feature_diff

def bias_amplification(train_captions=[],test_captions=[], train_dict_man = {}, train_dict_woman={}):
  # CALCULATING BIAS IN TRAIN CORPUS
  top_n_items(pos='noun', all_captions=train_captions)
  if train_captions!=[]: train_dict_man, train_dict_woman = bias()
  
  # CALCULATING BIAS IN TEST (GENERATED) CORPUS
  bias_items = [key for key,value in train_dict_man.items() if value>0.5]
  bias_items.extend([key for key,value in train_dict_woman.items() if value>0.5])
  print(bias_items)

  top_n_items(pos='noun', all_captions=test_captions)
  test_dict_man, test_dict_woman = bias()
  
  male_sum = bias_diff_sum(train_dict_man,test_dict_man)
  female_sum = bias_diff_sum(train_dict_woman, test_dict_woman)

  bias_amplification = male_sum + female_sum
  mean_bias_amplification = bias_amplification/len(bias_items)
  return mean_bias_amplification


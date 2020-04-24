# from biasanalysis import amp_utils,bias_analysis
from biasanalysis.amp_utils import *
from biasanalysis.bias_analysis import *
from utils import *

def return_biasamp(train_ids,pred_captions_dict):

	# loading training captions
	train_captions_dict = load_obj('captions_dict')

	train_captions = [caption for id in train_ids for caption in train_captions_dict[id]]
	
	test_captions = [caption for id in pred_captions_dict for caption in pred_captions_dict[id]]
	
	bias_amp = bias_amplification(train_captions, test_captions)
	
	return bias_amp

def return_bias(captions, gender='man'):
	top_n_items(pos='noun', all_captions=train_captions)
  	
	train_dict_man, train_dict_woman = bias()

	train_dict_man = {key:val for key,val in train_dict_man.items() if val>0.5 and val!=1}	
	train_dict_woman = {key:val for key,val in train_dict_woman.items() if val>0.5 and val!=1}	
	return train_dict_man if gender=='man' else train_dict_woman


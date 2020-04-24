# from biasanalysis import amp_utils,bias_analysis
from biasanalysis.amp_utils import *
from biasanalysis.bias_analysis import *
from utils import *

def return_bias(train_ids,pred_captions_dict):

	# loading training captions
	train_captions_dict = load_obj('captions_dict')

	train_captions = [caption for id in train_ids for caption in train_captions_dict[id]]
	
	test_captions = [caption for id in pred_captions_dict for caption in pred_captions_dict[id]]
	
	bias_amp = bias_amplification(train_captions, test_captions)
	
	return bias_amp

train_captions_dict = load_obj('captions_dict')

train_captions = [caption for id in list(train_captions_dict.keys())[:100] for caption in train_captions_dict[id]]

# return_bias([180521, 561629],{1005:['driving again car man'],3005:['woman being sassy glass','man holding plate','man holding plate','man holding plate']})
return_bias([180521, 561629],{1005:train_captions[:20],3005:train_captions[:20]})
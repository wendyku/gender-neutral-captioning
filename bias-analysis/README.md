This folder contains code for calculating gender bias scores and bias amplification in models. 
- bias_analysis.ipynb : Bias analyzed for 1000 randomly sampled images to obtain nouns and verbs most biased towards men/women. (midway report)
- training bias/ : contains saved training bias scores 
- activity_nouns, activity_verbs : contains balanced image id lists (activity and gender balanced) with 25 samples for each gender. Male, female and neutral genders considered. Lists are for 12 most biased nouns reported in bias_analysis.ipynb (based on sample availability).
- get_activitydata.py : code to generate the balanced image id lists




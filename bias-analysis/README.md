This folder contains code for calculating gender bias scores and bias amplification in models. 
- bias_analysis.ipynb : Bias analyzed for 1000 randomly sampled images to obtain nouns and verbs most biased towards (or against) men/women.
- training bias/ : contains sample bias scores
- activity_nouns, activity_verbs : contains balanced image id lists (activity and gender balanced) with 25 samples for each gender. Male, female and neutral genders considered. Lists are for each of the 6 most biased nouns in the randomly sampled training set (reported in bias_analysis.ipynb).
- get_activitydata.py : code to generate the balanced image id lists




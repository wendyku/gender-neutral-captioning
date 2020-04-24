# Correcting Gender Bias in Captioning Model

## Motivation
Our project discusses and tackles gender bias in captioning models, so that the image captions will be neutral of gender stereotypes. Specifically, this project 1) identifies types of gender bias in the captioning model, 2) experiments with different methods in reducing gender bias and 3) interprets the success of the final model. We also aim at building an inclusive captioning model that can distinguish not only the gender binary (ie. man or woman) but also a third category (ie. person) based on visual appearance.

## Environment and data set up
1. Install the project dependencies in your virtual environment. Refer to https://github.com/pypa/pipenv for commands specific to your OS. For macOS/ Linux, run
```
$ pipenv install --dev
```
Main dependencies:
- Python3
- torch
- torchvision
- nltk
- sklearn
- PIL
- matplotlib
- json

2. Clone this repository locally. Create a subdirectory `model`.
```
$ mkdir models
```
3. Download the following data from the COCO website(http://cocodataset.org/#download) into a subdirectory `data` located inside this project's directory. Move the folders into the repo's subdirectory `data`. This step is necessary only if intended to train a model or evaluate model results using COCO dataset.

- Under **Annotations**, download:
  - 2014 Train/Val annotations [241MB]
  
  By completion, the subdirectory `data/annotations/` should contain 5 files: `captions_train2014.json`, `captions_val2014.json`, `instanes_train2014.json` and `instances_val2014.json`.

- Under **Images**, download:
  - 2014 Train images [13GB]
  - 2014 Val images [6GB]
  
  By completion, the subdirectory `data/images/` should contain 2 folders, `train2014` and `val2014`.

3. Download folder containing pretained model (https://drive.google.com/open?id=1WLuLVc_57UgunkJmtlW78AeVZGWxVPEy). This model is trained on 4,625 COCO images with human figures as center of interest, using a balanced clean dataset and cross-entropy loss. Move the folders into the repo's main directory. This step is only necessary if intended to use pre-trained model.

- Download and unzip:
  - Gender_Neutral_Captioning_model
  
  By completion, the subdirectory `Gender_Neutral_Captioning_model/` should contain 3 files, `training_image_ids.pkl`, `vocab.pkl` and `best-model.pkl`.

## Run
The end-to-end process of our project can be reproduced via our Gender_Neutral_Captioning notebook. From the directory of the repo, run
```
$ jupyter notebook Gender_Neutral_Captioning.ipynb
```
The notebook consists of 4 major parts:
- Part I. Preparing Dataset for Training
- Part II. Model Training
  - a. Select method to generate training set
  - b. Train CNN+ LSTM model
- Part III. Predict on test images
  - a. Predict on human images in the COCO dataset
  - b. Predict on any images
- Part IV. Evaluate Model Performance

Note:
1. Each section can be run independently.
- To run an individual section, click the cell in the section and hit Cmd + enter. Alternatively, select Cell-> Run Cell from the top navigation bar.

2. All sections other than Part IIIb requires a full download of the COCO 2014 Training and Validation dataset, in the structure specified above.

## Happy captioning!

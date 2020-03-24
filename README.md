# Correcting Gender Bias in Captioning Model

## Motivation
Our project discusses and tackles gender bias in captioning models, so that the image captions will be neutral of gender stereotypes. Specifically, this project 1) identifies types of gender bias in the captioning model, 2) experiments with different methods in reducing gender bias and 3) interprets the success of the final model. We also aim at building an inclusive captioning model that can distinguish not only the gender binary (ie. man or woman) but also a third category (ie. person) based on visual appearance.

## Environment and Data Set up
1. Clone the COCO API Repo (https://github.com/cocodataset/cocoapi) and set up the API
```
git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI
make
cd ..
```
2. Install the project dependencies in your virtual environment. Refer to https://github.com/pypa/pipenv for commands specific to your OS. For macOS/ Linux, run
```
$ pipenv install --dev
```
3. Download the following data from the COCO website(http://cocodataset.org/#download) into the cocoapi subdirectory located inside this project's directory.

- Under **Annotations**, download:
  - 2014 Train/Val annotations [241MB]
  - 2014 Testing Image info [1MB]
  By completion, the subdirectory `cocoapi/annotations/` should contain 5 files: `captions_train2014.json`, `captions_val2014.json`, `instanes_train2014.json`, `instances_val2014.json`, `image_info_test2014.json`.

- Under **Images**, download:
  - 2014 Train images [13GB]
  - 2014 Val images [6GB]
  - 2014 Test images [6GB]
  By completion, the subdirectory `cocoapi/images/` should contain 3 folders, `train2014`, `val2014` and `test2014`.

4. Download pretrained ResNet-152 weights pretrained on Imagenet.

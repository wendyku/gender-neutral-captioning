#!/usr/bin/env python
# coding: utf-8



from bias_analysis import *
from collections import defaultdict
import glob

# top 12 biased verbs from both genders
bias_activities = ('driving','passing','catching','reading','touching','pulling','wedding','jumping','skateboarding','pulled','riding',
                   'pushing','cutting','having','laying','cut','petting','waiting','talking','haired','overlooking','staring','typing')

# top 12 biased nouns from both genders
bias_items = ('tennis','dirt','skateboard','grass','carriage','beer','kite','beach','road','wagon','bags','surfboard',
              'dress','lap','bed','flowers','mouth','bridle','fire','teeth','curb','device','leash','face')


with open('female_word_file.txt') as f:
    female_nouns = [line[:-1] for line in f]
with open('male_word_file.txt') as f:
    male_nouns = [line[:-1] for line in f]



# Get 25 samples of each gender doing activity (ie. total 50 samples)
def get_activity_ids(ids, gender, bias_items, n_samples):
    gender_nouns = male_nouns if gender=='man' else female_nouns
    activity_ids_dict = defaultdict(list)
    captions_dict = defaultdict(list)
    print("Getting samples: ",n_samples, "nos.")
       
    with open('captions_train2014.json') as file:
        coco_data = json.load(file)

    for item in coco_data['annotations']:
        bias_items = [i for i in bias_items if len(activity_ids_dict[i])<n_samples]
        
        # if all n_samples collected for all bias_items, break
        if bias_items==[]:
            break

        # going through each annotation
        if item['image_id'] in ids:
            for activity in bias_items:
                if any(word in gender_nouns for word in item['caption'].split()) and (activity in item['caption'].split()):
                    if item['image_id'] not in activity_ids_dict[activity]:      
                        activity_ids_dict[activity].append(item['image_id'])  
                        captions_dict[activity].append(item['caption'])  

    return activity_ids_dict,captions_dict



def get_allids():
    with open('instances_train2014.json') as file:
        data = json.load(file)
    all_ids = []
    for item in data['annotations']:
        if item['category_id'] == 1:
            all_ids.append(item['image_id'])
    print("Got All IDS")
    return all_ids




n_samples = 25
all_ids = get_allids()
for bias_list in [bias_items,bias_activities]:
    
    for gender in ['man','woman']:
        ids, captions = get_activity_ids(all_ids,gender,bias_list,n_samples)
        for activity in ids:
            print(activity, len(ids[activity]))
            with open(activity+".txt",'a') as out:
                if bias_list == bias_activities: out.write("\n")
                write_ids = [str(i) for i in ids[activity]]
                out.write("\n".join(write_ids))
    

# Getting images and copying to folder
filenames = []
with open('captions_val2014.json') as file:
    coco_data = json.load(file)
for item in coco_data['images']:
    if item['id'] in ids['tennis']:
        filenames.append(item['file_name'])



target_folder='images' #target_folder=os.mkdir('images')
for image in filenames:
    image = "val2014/" + image 
    shutil.copy(image, target_folder)
print("Copied sampled images to {}".format(target_folder))



activity_list_paths = glob.glob("*.txt")           

os.mkdir('activity_nouns')
os.mkdir('activity_verbs')
for file in activity_list_paths:
    with open(file) as f:
        length = sum([1 for line in f])
    if (length==50) and file[:-4] in bias_activities: 
        try:
            shutil.move(file, 'activity_verbs')
        except:print()
    if (length==50) and file[:-4] in bias_items: 
        try:
            shutil.move(file, 'activity_nouns')
        except:print()



activity_list_paths = glob.glob("activity_verbs/*.txt")  

files = [item.split('\\')[1] for item in activity_list_paths]
for file,name in zip(activity_list_paths,files):
    new_name = "intersection_"+name[:-4]+"_person.txt"
    os.rename(file,new_name)






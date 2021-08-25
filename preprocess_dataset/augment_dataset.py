# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

import numpy as np
import random
import pickle
import os, os.path
import argparse
import sys


def get_str_class(array):
    if len(array)>8:
        array=np.fromstring(array, dtype=int, sep=' ')
        converted_class = array
    elif isinstance(array[0], str):
        converted_class = []
        for i in range(len(array)):
            converted_class.append(int(array[i]))
        converted_class = np.array(converted_class)
    else:
        converted_class = array

    # print(array)
    # print(np.array_str(converted_class)[1:-1])
    return np.array_str(converted_class)[1:-1]

"""
Modifies the pkl dataset given raw data into the format: [img1, img2, action, action spec, class1, class2]'
It does not consider boxes color
a = binary classification if action took place
"""
def main(args):
    print('Args: '+str(args))
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dataset', type=str, required=False, default='datasets/2500_shelf_stacking_all_distractors',
                        help='dataset')
    parser.add_argument('--ac_rand_mul', type=int, required=False, default=2,
                        help='dataset')
    args_in = parser.parse_args(args)
    pkl_dataset = args_in.pkl_dataset
    rand_mul = args_in.ac_rand_mul

    print("Building dataset with action random factor = "+str(rand_mul))
    random.seed(1)

    #Load dataset
    with open(pkl_dataset+".pkl", 'rb') as f:
        or_list = pickle.load(f)
        print("Pkl dataset: "+str(pkl_dataset))

    #Create lists
    action_list = []
    no_action_list = []
    merged_class_dict = {}
    noaction_class_dict = {}
    all_classes_str=[]
    all_classes=[]
    for item in or_list:
        all_classes.append(item[3])
        all_classes.append(item[4])
        class_1 = get_str_class(item[-2])
        class_2 = get_str_class(item[-1])

        if item[2] == 1:
            action_list.append(item) #action pairs
        else:
            no_action_list.append(item)

            if class_1 in noaction_class_dict.keys():
                noaction_class_dict[class_1] = noaction_class_dict[class_1] + 1
            else:
                noaction_class_dict[class_1] = 1


        all_classes_str.append(class_1)
        all_classes_str.append(class_2)
        if class_1 in merged_class_dict.keys():
            merged_class_dict[class_1].append(item[0])
        else:
            merged_class_dict[class_1] = [item[0]]


        if class_2 in merged_class_dict.keys():
            merged_class_dict[class_2].append(item[1])
        else:
            merged_class_dict[class_2] = [item[1]]

    all_classes_str=np.unique(np.array(all_classes_str),axis=0)
    all_classes=np.unique(np.array(all_classes),axis=0)
    all_classes_str.sort()
    all_classes.sort()
    print("No action classes: "+str(len(noaction_class_dict)))
    print("Action pairs: "+str(len(action_list)))
    print("No action pairs: "+str(len(no_action_list)))
    print("Classes: "+str(len(merged_class_dict)))

    #print(merged_class_dict.keys())
    first_class = list(merged_class_dict.keys())[0]

    list_keys = []
    for key in merged_class_dict.keys():
        list_keys.append(key)
    list_keys = sorted(list_keys)

    # Create new action pairs
    new_action_list = []
    for item in action_list:
        class_1 = get_str_class(item[-2])
        class_2 = get_str_class(item[-1])
        #Append action pair
        new_action_list.append((item[0], item[1], item[2], class_1, class_2))

        #Generate random pairs
        for i in range(rand_mul-1):
            random_key = random.choice(list_keys)
            random_img2 = random.choice(merged_class_dict[random_key])
            # the first image is the same as before, the second is randomly selected
            new_action_list.append((item[0], random_img2, item[2], class_1, random_key))

    
    new_no_action_list = no_action_list


    pkl_list = new_no_action_list[:] + new_action_list[:]
    random.shuffle(pkl_list)

    print('_________________________________')
    print('Total pairs: '+str(len(pkl_list)))
    #
    with open(pkl_dataset+"_ac_random_"+str(rand_mul) +".pkl", 'wb') as f:
        pickle.dump(pkl_list, f)



if __name__ == "__main__":

    main(sys.argv[1:])

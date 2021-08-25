# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

from __future__ import print_function
import argparse
import os
import sys
import pickle
import torch
import time
import numpy as np
import algorithms as alg
from importlib.machinery import SourceFileLoader
from utils import t_snes_plot,nice_colors,make_nice_labels



#compute mean and std of action or no action pairs
def compute_mean_and_std_dev(latent_map_file,distance_type,action_mode=0):

    f = open(latent_map_file, 'rb')
    latent_map = pickle.load(f)
    len_latent_map = len(latent_map)


    distance_type=1
    dist_list = []

    for latent_pair in latent_map:
        # get the latent coordinates
        z_pos_c1=latent_pair[0]
        z_pos_c2=latent_pair[1]
        action=latent_pair[2]
        
        if action_mode==0:
            if action == 0:
                current_distance=np.linalg.norm(z_pos_c1-z_pos_c2, ord=distance_type)
                dist_list.append(current_distance)
        if action_mode==1:
            if action == 1:
                current_distance=np.linalg.norm(z_pos_c1-z_pos_c2, ord=distance_type)
                dist_list.append(current_distance)


    mean_dist_no_ac = np.mean(dist_list)
    std_dist_no_ac = np.std(dist_list)
    return mean_dist_no_ac, std_dist_no_ac, dist_list



#Label latent space given VAE and dataset
def lable_latent_space(config_file,checkpoint_file,output_file,dataset_name):

    #load model
    vae_config_file = os.path.join('.', 'configs', config_file + '.py')
    vae_directory = os.path.join('.', 'models', checkpoint_file)
    vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config
    vae_config['exp_name'] = config_file
    vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
    #print(' *- Loading config %s from file: %s' % (config_file, vae_config_file))
    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    #print(' *- Loaded {0}'.format(vae_config['algorithm_type']))
    num_workers = 1#vae_config_file['vae_opt']['num_workers']
    data_test_opt = vae_config['data_train_opt']

    f = open('datasets/'+dataset_name+'.pkl', 'rb')
    dataset = pickle.load(f)


    vae_algorithm.load_checkpoint('models/'+config_file+"/"+checkpoint_file)
    vae_algorithm.model.eval()


    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_map=[]
    for i in range(len(dataset)):
        t=dataset[i]
        img1 = torch.tensor(t[0]/255.).float().permute(2, 0, 1)
        img2 = torch.tensor(t[1]/255).float().permute(2, 0, 1)
        img1=img1.unsqueeze_(0)
        img2=img2.unsqueeze_(0)

        a=t[2]
        a_lambda=-1
        class_1=t[3]
        class_2=t[4]
        #data in pkl file are images filename. In order for the VAE to work, the images must be converted in RGB and converted into range [0,1]
        img1 = img1.to(device)
        img2 = img2.to(device)


        if vae_config['algorithm_type'] == "AE_Algorithm":
            dec_mean1,  z =vae_algorithm.model.forward(img1)
            dec_mean2,  z2=vae_algorithm.model.forward(img2)
        else:
            dec_mean1, dec_logvar1, z, enc_logvar1=vae_algorithm.model.forward(img1)
            dec_mean2, dec_logvar2, z2, enc_logvar2=vae_algorithm.model.forward(img2)


        for i in range(z.size()[0]):
            z_np=z[i,:].cpu().detach().numpy()
            z2_np=z2[i,:].cpu().detach().numpy()
            latent_map.append((z_np,z2_np,a,class_1,class_2))  

    #dump pickle
    with open(output_file+'.pkl', 'wb') as f:
        pickle.dump(latent_map, f)



def main():

    all_configs=[]

    #AE model-Box
    box_ae_configs=[ [ "ae_box_stacking_4b_vp_view_0_no_color_baseline","box_stacking_4b_vp_view_0_holdout_no_color"],
                   [ "ae_box_stacking_4b_vp_view_1_no_color_baseline","box_stacking_4b_vp_view_1_holdout_no_color" ],
                   [ "ae_box_stacking_4b_vp_view_2_no_color_baseline","box_stacking_4b_vp_view_2_holdout_no_color"],
                   [ "ae_box_stacking_4b_vp_view_hard_mixed_no_color_baseline", "box_stacking_4b_vp_view_hard_mixed_holdout_no_color" ]
                ]

    #PC-AE model-Box
    box_pc_ae_configs=[ [ "ae_box_stacking_4b_vp_view_0_no_color_w1000","box_stacking_4b_vp_view_0_holdout_no_color"],
                   [ "ae_box_stacking_4b_vp_view_1_no_color_w1000","box_stacking_4b_vp_view_1_holdout_no_color" ],
                   [ "ae_box_stacking_4b_vp_view_2_no_color_w1000","box_stacking_4b_vp_view_2_holdout_no_color"],
                   [ "ae_box_stacking_4b_vp_view_hard_mixed_no_color_w1000", "box_stacking_4b_vp_view_hard_mixed_holdout_no_color" ]
                   ]


    #VAE model-Box
    box_vae_configs=[ [ "iros_vae_box_stacking_4b_vp_view_0_no_color_baseline","box_stacking_4b_vp_view_0_holdout_no_color"],
                   [ "iros_vae_box_stacking_4b_vp_view_1_no_color_baseline","box_stacking_4b_vp_view_1_holdout_no_color" ],
                   [ "iros_vae_box_stacking_4b_vp_view_2_no_color_baseline","box_stacking_4b_vp_view_2_holdout_no_color"],
                   [ "iros_vae_box_stacking_4b_vp_view_hard_mixed_no_color_baseline", "box_stacking_4b_vp_view_hard_mixed_holdout_no_color" ]
                ]

    #PC-VAE model-Box
    box_pc_vae_configs=[    [ "iros_vae_box_stacking_4b_vp_view_0_no_color_w1000","box_stacking_4b_vp_view_0_holdout_no_color"],
                   [ "iros_vae_box_stacking_4b_vp_view_1_no_color_w1000","box_stacking_4b_vp_view_1_holdout_no_color" ],
                   [ "iros_vae_box_stacking_4b_vp_view_2_no_color_w1000","box_stacking_4b_vp_view_2_holdout_no_color"],
                   [ "iros_vae_box_stacking_4b_vp_view_hard_mixed_no_color_w1000", "box_stacking_4b_vp_view_hard_mixed_holdout_no_color" ]
                   ]
    

   #-------------------------------------------------------------------------------------------------------------------------------------


    #AE- model shelf
    shelf_ae_configs=[ [ "ae_shelf_stacking_baseline", "2500_shelf_stacking_holdout" ],
                       [ "ae_shelf_stacking_ac_random_2_baseline","2500_shelf_stacking_holdout" ],
                       #-------------------------------------------------------------------------------------------
                       [ "ae_shelf_stacking_all_distractors_baseline", "2500_shelf_stacking_holdout" ],
                       [ "ae_shelf_stacking_all_distractors_ac_random_2_baseline","2500_shelf_stacking_holdout" ]
                      ]

    #PC-AE model shelf
    shelf_pc_ae_configs=[ [ "ae_shelf_stacking", "2500_shelf_stacking_holdout" ],
                       [ "ae_shelf_stacking_ac_random_2","2500_shelf_stacking_holdout" ],
                       #-------------------------------------------------------------------------------------------
                       [ "ae_shelf_stacking_all_distractors__w1000", "2500_shelf_stacking_holdout" ],
                       [ "ae_shelf_stacking_all_distractors_ac_random_2","2500_shelf_stacking_holdout" ]
                      ]

    #VAE- model shelf
    shelf_vae_configs=[ [ "iros_vae_shelf_stacking_baseline", "2500_shelf_stacking_holdout" ],
                       [ "iros_vae_shelf_stacking_ac_random_2_baseline","2500_shelf_stacking_holdout" ]
                       #-------------------------------------------------------------------------------------------
                       [ "iros_vae_shelf_stacking_all_distractors_baseline", "2500_shelf_stacking_holdout" ],
                       [ "iros_vae_shelf_stacking_all_distractors_ac_random_2_baseline","2500_shelf_stacking_holdout" ]
                      ]

    #PC-VAE ,odel shelf
    shelf_pc_vae_configs=[ [ "iros_vae_shelf_stacking", "2500_shelf_stacking_holdout" ],
                       [ "iros_vae_shelf_stacking_ac_random_2_noac_swaps_0","2500_shelf_stacking_holdout" ]
                       #-------------------------------------------------------------------------------------------
                       [ "iros_vae_shelf_stacking_all_distractors", "2500_shelf_stacking_holdout" ],
                       [ "iros_vae_shelf_stacking_all_distractors_ac_random_2","2500_shelf_stacking_holdout" ]
                      ]

    #----------------------------------------------------------------------------------------------------------------------

    #PC-AE- model box manipulation
    box_manip_ae_configs=[ [ "ae_box_manipulation_view1_train_baseline","box_manipulation_view1_holdout"],
                   [ "ae_box_manipulation_view2_train_baseline","box_manipulation_view2_holdout" ],
                   [ "ae_box_manipulation_view1_view2_train_baseline","box_manipulation_view1_view2_holdout"],
                   [ "ae_box_manipulation_view1_train_ac_random_2_baseline","box_manipulation_view1_holdout"],
                   [ "ae_box_manipulation_view2_train_ac_random_2_baseline","box_manipulation_view2_holdout" ],
                   [ "ae_box_manipulation_view1_view2_train_ac_random_2_baseline","box_manipulation_view1_view2_holdout"]
                ]

    #PC-AE- model box manipulation
    box_manip_pc_ae_configs=[ [ "ae_box_manipulation_view1_train","box_manipulation_view1_holdout"],
                   [ "ae_box_manipulation_view2_train","box_manipulation_view2_holdout" ],
                   [ "ae_box_manipulation_view1_view2_train","box_manipulation_view1_view2_holdout"],
                   [ "ae_box_manipulation_view1_train_ac_random_2","box_manipulation_view1_holdout"],
                   [ "ae_box_manipulation_view2_train_ac_random_2","box_manipulation_view2_holdout" ],
                   [ "ae_box_manipulation_view1_view2_train_ac_random_2","box_manipulation_view1_view2_holdout"]
                ]

    #VAE- model box manipulation
    box_manip_vae_configs=[ [ "iros_vae_box_manipulation_view1_train_baseline","box_manipulation_view1_holdout"],
                   [ "iros_vae_box_manipulation_view2_train_baseline","box_manipulation_view2_holdout" ],
                   [ "iros_vae_box_manipulation_view1_view2_train_baseline","box_manipulation_view1_view2_holdout"],
                   [ "iros_vae_box_manipulation_view1_train_ac_random_2_baseline","box_manipulation_view1_holdout"],
                   [ "iros_vae_box_manipulation_view2_train_ac_random_2_baseline","box_manipulation_view2_holdout" ],
                   [ "iros_vae_box_manipulation_view1_view2_train_ac_random_2_baseline","box_manipulation_view1_view2_holdout"]
                ]

    #PC-VAE- model box manipulation
    box_manip_pc_vae_configs=[ [ "iros_vae_box_manipulation_view1_train","box_manipulation_view1_holdout"],
                   [ "iros_vae_box_manipulation_view2_train","box_manipulation_view2_holdout" ],
                   [ "iros_vae_box_manipulation_view1_view2_train","box_manipulation_view1_view2_holdout"],
                   [ "iros_vae_box_manipulation_view1_train_ac_random_2","box_manipulation_view1_holdout"],
                   [ "iros_vae_box_manipulation_view2_train_ac_random_2","box_manipulation_view2_holdout" ],
                   [ "iros_vae_box_manipulation_view1_view2_train_ac_random_2","box_manipulation_view1_view2_holdout"]
                ]


    #----------------------------------------------------------------------------------------------------------------------

    all_configs.append(box_ae_configs)
    all_configs.append(box_pc_ae_configs)
    all_configs.append(box_vae_configs)
    all_configs.append(box_pc_vae_configs)
    all_configs.append(shelf_ae_configs)
    all_configs.append(shelf_pc_ae_configs)
    all_configs.append(shelf_vae_configs)
    all_configs.append(shelf_pc_vae_configs)
    all_configs.append(box_manip_ae_configs)
    all_configs.append(box_manip_pc_ae_configs)
    all_configs.append(box_manip_vae_configs)
    all_configs.append(box_manip_pc_vae_configs)
    
    checkpoint_file="vae_lastCheckpoint.pth"

    for configs in all_configs:
      for entry in configs:
          config_file=entry[0]
          dataset_name=entry[1]

          output_file="models/"+ config_file +"/"+config_file+ "_z_encodings"+dataset_name

          lable_latent_space(config_file,checkpoint_file,output_file,dataset_name)
          print("map done")
          time.sleep(2)
          mean_dist_no_ac, std_dist_no_ac, dist_list=compute_mean_and_std_dev(output_file+'.pkl','1',action_mode=1)

          latent_map = pickle.load( open( output_file+'.pkl', "rb" ) )

          t_snes_plot(latent_map,config_file,dataset_name)

          f = open("balsine_distances.txt", "a")
          print(mean_dist_no_ac)
          w_string=config_file+ " , mean action dis: " + str(mean_dist_no_ac) + "\n"
          f.write(w_string)
          f.close()

if __name__== "__main__":
  main()

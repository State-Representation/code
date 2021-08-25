#!/bin/bash 
# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

#Box stacking
#python train_recon_models.py --exp_vae='ae_box_stacking_4b_vp_view_0_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_stacking_4b_vp_view_1_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_stacking_4b_vp_view_2_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_stacking_4b_vp_view_hard_mixed_no_color_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='ae_box_stacking_4b_vp_view_0_no_color_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_stacking_4b_vp_view_1_no_color_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_stacking_4b_vp_view_2_no_color_w1000' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_stacking_4b_vp_view_hard_mixed_no_color_w1000' --cuda=True 
#
#python train_recon_models.py --exp_vae='iros_vae_box_stacking_4b_vp_view_0_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_stacking_4b_vp_view_1_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_stacking_4b_vp_view_2_no_color_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_stacking_4b_vp_view_hard_mixed_no_color_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='iros_vae_box_stacking_4b_vp_view_0_no_color' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_stacking_4b_vp_view_1_no_color' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_stacking_4b_vp_view_2_no_color' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_stacking_4b_vp_view_hard_mixed_no_color' --cuda=True 
#
#
##shelf arrangment
#python train_recon_models.py --exp_vae='ae_shelf_stacking_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_shelf_stacking_ac_random_2_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='ae_shelf_stacking_all_distractors_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_shelf_stacking_all_distractors_ac_random_2_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='ae_shelf_stacking' --cuda=True 
#python train_recon_models.py --exp_vae='ae_shelf_stacking_ac_random_2' --cuda=True 
#
#python train_recon_models.py --exp_vae='ae_shelf_stacking_all_distractors' --cuda=True 
#python train_recon_models.py --exp_vae='ae_shelf_stacking_all_distractors_ac_random_2' --cuda=True 
#
#
#python train_recon_models.py --exp_vae='iros_vae_shelf_stacking_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_shelf_stacking_ac_random_2_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='iros_vae_shelf_stacking_all_distractors_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_shelf_stacking_all_distractors_ac_random_2_baseline' --cuda=True
#
#python train_recon_models.py --exp_vae='iros_vae_shelf_stacking' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_shelf_stacking_ac_random_2' --cuda=True 
#
#python train_recon_models.py --exp_vae='iros_vae_shelf_stacking_all_distractors' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_shelf_stacking_all_distractors_ac_random_2' --cuda=True
#
#
##box manipulation
#python train_recon_models.py --exp_vae='ae_box_manipulation_view1_train_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_manipulation_view2_train_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_manipulation_view1_view2_train_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='ae_box_manipulation_view1_train_ac_random_2_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_manipulation_view2_train_ac_random_2_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_manipulation_view1_view2_train_ac_random_2_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='ae_box_manipulation_view1_train' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_manipulation_view2_train' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_manipulation_view1_view2_train' --cuda=True 
#
#python train_recon_models.py --exp_vae='ae_box_manipulation_view1_train_ac_random_2' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_manipulation_view2_train_ac_random_2' --cuda=True 
#python train_recon_models.py --exp_vae='ae_box_manipulation_view1_view2_train_ac_random_2' --cuda=True 
#
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view1_train_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view2_train_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view1_view2_train_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view1_train_ac_random_2_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view2_train_ac_random_2_baseline' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view1_view2_train_ac_random_2_baseline' --cuda=True 
#
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view1_train' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view2_train' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view1_view2_train' --cuda=True 
#
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view1_train_ac_random_2' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view2_train_ac_random_2' --cuda=True 
#python train_recon_models.py --exp_vae='iros_vae_box_manipulation_view1_view2_train_ac_random_2' --cuda=True 

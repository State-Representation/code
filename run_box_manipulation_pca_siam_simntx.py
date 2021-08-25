# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

from utils import train_siamese, train_simntx, train_pca
from utils import infer_siamese, infer_simntx, infer_pca

#produce shelf arrangment modles
prefix="shelf_arrangment"

models=["siamese","simntx","pca"]

datasets=[ ["box_manipulation_view1_train", ["box_manipulation_view1_train", "box_manipulation_view1_holdout"]],
				 ["box_manipulation_view2_train", ["box_manipulation_view2_train", "box_manipulation_view2_holdout"]],
				 ["box_manipulation_view1_view2_train", ["box_manipulation_view1_view2_train", "box_manipulation_view1_view2_holdout"]],
				 ["box_manipulation_view1_train_ac_random_2", ["box_manipulation_view1_train", "box_manipulation_view1_holdout"]],
				 ["box_manipulation_view2_train_ac_random_2", ["box_manipulation_view2_train", "box_manipulation_view2_holdout"]],
				 ["box_manipulation_view1_view2_train_ac_random_2", ["box_manipulation_view1_view2_train", "box_manipulation_view1_view2_holdout"]]
				]

#training
for dataset in datasets:
	for model in models:	
		train_dataset=dataset[0]
		model_name=prefix + model + "_" + train_dataset
		if model=="siamese":
			train_siamese(model_name,train_dataset)
		if model=="simntx":
			train_simntx(model_name,train_dataset)
		if model=="pca":
			train_pca(model_name,train_dataset)

#inference
	for model in models:
		infer_datasets=dataset[1]
		model_name=prefix + model+ "_"  + train_dataset
		for infer_dataset in infer_datasets:		
			if model=="siamese":
				infer_siamese(model_name,infer_dataset)
			if model=="simntx":
				infer_simntx(model_name,infer_dataset)
			if model=="pca":
				infer_pca(model_name,infer_dataset)


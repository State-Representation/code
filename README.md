# Comparing Reconstruction- and Contrastive-based Models for Visual Task Plannin
This is the code for the paper: Comparing Reconstruction- and Contrastive-based Models for Visual Task Plannin

Visit the dedicated website: [here](https://github.com/State-Representation/web) for more information.

If you use this code in your work, please cite it as follows:


## Bibtex

```
Avalible after Puplication
```

## Training the models

### setup

```
pip install -r requirements.txt
```

### Datasets
prepare datasets:
```
cd datasets
chmod +x get_datasets.sh
./get_datasets.sh
```

### Augment shelf Datasets
prepare datasets:
```
python preprocess_dataset/augment_dataset.py --pkl_dataset='datasets/2500_shelf_stacking' --ac_rand_mul=2
python preprocess_dataset/augment_dataset.py --pkl_dataset='datasets/2500_shelf_stacking_all_distractors' --ac_rand_mul=2
```

### Augment box manipulation Datasets
```
python preprocess_dataset/augment_dataset.py --pkl_dataset='datasets/box_manipulation_view1_train' --ac_rand_mul=2
python preprocess_dataset/augment_dataset.py --pkl_dataset='datasets/box_manipulation_view2_train' --ac_rand_mul=2
python preprocess_dataset/augment_dataset.py --pkl_dataset='datasets/box_manipulation_view1_view2_train' --ac_rand_mul=2
```



### Train PCA; PC-Sia.; CE-Sia. Models
Box stacking task:

```
python run_box_stacking_pca_siam_simntx.py
```

Shelf arranging task:
```
python run_shelf_arrangment_pca_siam_simntx.py
```

Box manipulation task:
```
python run_box_manipulation_pca_siam_simntx.py
```

### Train AE; PC-AE; VAE; PC-VAE

Builds on code work from: [Latent Space Roadmap website](https://visual-action-planning.github.io/lsr/)
```
python prepare_recon_dataset_for_cluster.py
```

```
chmod +x run_train_recon_models.sh
./run_train_recon_models.sh
```

### Inference AE; PC-AE; VAE; PC-VAE
To get the encodins and plots run:

```
python produce_recon_models_encodings.py
```


### Results

The representations are saved in the model folder together with the t-SNE plots:

![plots example](box_stacking.png)




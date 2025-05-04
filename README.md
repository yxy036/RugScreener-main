# RugScreener

Official code for RugScreener: Leveraging Temporal Graph Neural Network for Rugpull Detection in Decentralized Finance.

For ease of use, we have made some changes to the original implementation in the paper.

---

## Running the experiments

### Requirements

Dependencies (with python >= 3.6):

```{bash}
pandas==1.1.5
torch==1.9.0+cu102
torchvision==0.10.0+cu102
dgl==0.8.1+cu102
scikit_learn==0.24.2
```

### Dataset preprocessing

The load_data.py file is responsible for preprocessing the dataset, but it requires modifications to the code based on the file name and dataset details during execution.

```{bash}
python load_data.py
```

### Model training

The train.py file trains the model using the training dataset and saves the trained model.

The omitted parameters will use the predefined default values, which can be found in the train.py file.

```{bash}
# Train the model
python train.py --dataset [train_dataset] --simple_mode --k_hop [sampling k-hop neighborhood] --embedding_dim [embedding dim] --memory_dim [memory dim] --temporal_dim [temporal dimension] --n_neighbors [number of neighbors] --num_heads [num_heads for multihead attention] --file [log_filename] --save [model_savepath]
# Example
python train.py --dataset data_std --simple_mode --k_hop 1 --file log_std_tgn.txt --save std_tgn

# If you want to use TGAT
python train.py --dataset [train_dataset] --simple_mode --not_use_memory --k_hop 2 --file [log_filename] --save [model_savepath]
# Example
python train.py --dataset data_std --simple_mode --not_use_memory --k_hop 2 --file log_std_tgat.txt --save std_tgat
```

### Model testing

The classification.py file loads the trained model to perform classification. To facilitate future testing, we save the classification sample files learned through the model.

```{bash}
python classification.py --dataset [test_dataset] --simple_mode --save_path [trained_model_savepath]
# Example
python classification.py --dataset data_std_label --simple_mode --save_path ./model/std_tgn/trained_model.pkl
```

If a saved classification sample file already exists, we use the load_classify.py file to load the saved classification sample file and call the classification model for classification.

```{bash}
python load_classify.py 
```

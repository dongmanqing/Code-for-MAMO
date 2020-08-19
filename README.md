# Code-for-MAMO
Code for paper MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation. 


## Requirements 
- python 3.6+

Packages
- pytorch
- numpy
- pandas

## Dataset

1. The raw datasets could be downloaded from: 
- [MovieLens](https://grouplens.org/datasets/movielens/)
- [Bookcrossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

2. Put the dataset into the according `data_raw` folder. 

3. Create a folder named `data_processed`, you can process the raw datasets via
`python3 prepareDataset.py`

Here we only give the processing code for the MovieLens dataset, please write your own code for processing Bookcrossing dataset with the similar functions presented in `prepareMovielens.py`

4. The structure of the processed dataset:

```
- data_processed

  - bookcrossing
    - raw
      sample_1_x1.p
      sample_1_x2.p
      sample_1_y.p
      sample_1_y0.p
      ...
    item_dict.p
    item_state_ids.p
    ratings_sorted.p
    user_dict.p
    user_state_ids.p
   
  - movielens
    - raw
      sample_1_x1.p
      sample_1_x2.p
      sample_1_y.p
      sample_1_y0.p
      ...
    item_dict.p
    item_state_ids.p
    ratings_sorted.p
    user_dict.p
    user_state_ids.p
```

## Model training
The structure of our code: 
```
- prepare_data
  prepareBookcrossing.py
  prepareList.py
  prepareMovielens.py
- modules
  info_embedding.py
  input_loading.py
  memories.py
  rec_model.py
configs.py
mamoRec.py
models.py
prepareDataset.py
utils.py
```

Run the codes in `mamoRec.py` for training the model:
```
if __name__ == '__main__':
    MAMRec('movielens')
```

## Some tips
1. This version of code runs over all training or testing users, which may take about half an hour for one epoch on a Linux server with NVIDIA TITAN X. So you can revise the code for updating the parameters via batches of users and using parallel computing. 

2. You can del the used variables to save the computation cost. If you have any suggestions on saving the computation cost, I'm happy to receive your emails. 


## Citation 
If you use this code, please consider to cite the following paper:

```
@inproceedings{dong2020mamo,
  title={MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation},
  author={Manqing, Dong and Feng, Yuan and Lina, Yao and Xiwei, Xu and Liming, Zhu},
  booktitle={26th SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2020}
}
```

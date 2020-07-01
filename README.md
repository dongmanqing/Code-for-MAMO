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

3. You can either process the raw datasets via
`python3 prepareDataset.py`
or use [our processed data](https://drive.google.com/file/d/1t9I01YtBc-5d3bGJf2UAd_J3FNIDvpux/view?usp=sharing). 

Here we give the processing code for the MovieLens dataset, if you also want the code for processing the Bookcrossing dataset, please star this project and email me :). 

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

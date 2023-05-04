# Introduction

This repository is an example of paper "Sample Efficient Reinforcement Learning Using Graph-Based Memory Reconstruction" in  Tensorflow.

```
@ARTICLE{10105983,
  author={Kang, Yongxin and Zhao, Enmin and Zang, Yifan and Li, Lijuan and Li, Kai and Tao, Pin and Xing, Junliang},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Sample Efficient Reinforcement Learning Using Graph-Based Memory Reconstruction}, 
  year={2023},
  pages={1-12},
  doi={10.1109/TAI.2023.3268612}
  }
```

# Training

The following command runs GBMR in  Montezuma's Revenge:

```
cd GBMR
CUDA_VISIBLE_DEVICES=7 python mainGBMRVA.py --env=ENVname --training_iters=10000000 --memory_size=500000 --epsilon=0.1 --display_step=10000 --learn_step=4 --num_neighbours=50 --dist_th=2 --riqi="0518" --expert_memory_size=10 --save_path="/home/kpl/SEGBMRresults/GBMRVA"
```

You can select "--env ENVname" to run in different games, eg. --env="Alien-v4"


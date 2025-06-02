# Group-On: Boosting One-Shot Segmentation with  Supportive Query
Group-On is , as introduced in [ICME 2025 oral paper](https://arxiv.org/abs/2404.11871). Please refer to the [arxiv version](https://arxiv.org/abs/2404.11871) for more technical details.

<img src="assets/framework.png" alt="framework" width="850" height="300"> 
Figure 1: Illustrating GROUP-ON in the one-shot setting. 
Different from the conventional few-shot setting, a group of query images along with one support image is input into the model. After the coarse masks of all the query images are segmented, the supportive image-mask pairs function as pseudo support to segment the host query again. Eventually, the final result is produced by the proposed MoME module, where a flexible number of mask experts make decisions on candidate masks guided by a scene-driven router. Note that the host and supportive queries play an equal and interchangeable role.

## Requirements
Details are in [requirements](). Here is the basic setting:
```
Python=3.10
PyTorch=1.13.1
cuda=11.7
```

## License
This codebase is released under the Apache License 2.0 as in the LICENSE file. 

## Citation 
If you find this research work interesting and helpful, please cite our paper:
```
@inproceedings{zhou2025protclip,
  title={ProtCLIP: Function-Informed Protein Multi-Modal Learning},
  author={Zhou, Hanjing and Yin, Mingze and Chen, Danny and Wu, Jian and Chen, Jintai},
  booktitle={International Conference on Multimedia and Expo},
  volume={0},
  number={0},
  pages={0--0},
  year={2025}
}
```
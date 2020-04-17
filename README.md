# ChipGAN
ChipGAN for Chinese Ink Wash Painting Style Transfer

## Training and Testing 
1. The code is based on python2.7 + pytorch3.0. 
2. The two .sh files are used to train and test the model.
3. You can modify the "train_horse2_edge_10_dec_150.sh" for other stylization tasks.
4. The pretrained model is  in the checkpoints file.
5. You can directly run "bash test_horse2_10_dec_150.sh" to get the stylished images in the "results" file.

## CHIPPHI Dataset
1. You can download our dataset from https://pan.baidu.com/s/1oXFVv1tZCkUSoH2pSxWFSA with password `nqhi`

## Citation
```
@inproceedings{10.1145/3240508.3240655,
author = {He, Bin and Gao, Feng and Ma, Daiqian and Shi, Boxin and Duan, Ling-Yu},
title = {ChipGAN: A Generative Adversarial Network for Chinese Ink Wash Painting Style Transfer},
year = {2018},
isbn = {9781450356657},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3240508.3240655},
doi = {10.1145/3240508.3240655},
booktitle = {Proceedings of the 26th ACM International Conference on Multimedia},
pages = {1172–1180},
numpages = {9},
keywords = {generative adversarial network, style transfer, painting},
location = {Seoul, Republic of Korea},
series = {MM ’18}
}
```

## Contactor
If you have any question, please feel free to contact me with cs_hebin@pku.edu.cn

# Weighted Feature Fusion of Convolutional Neural Network and Graph Attention Network for Hyperspectral Image Classification

This example implements the paper [Weighted Feature Fusion of Convolutional Neural Network and Graph Attention Network for Hyperspectral Image Classification]

## Run
If you want to run this code, just put your data in the Datasets folder and change a few paths.
- path 1: datasets: Please put the corresponding hyperspectral data there.
- path 2: loadData/data_reader.py: change datasets path.

> python main.py -pc -pdi -sr

## Installation
This project is implemented with Pytorch and has been tested on version 
- Pytorch               1.7, 
- numpy                 1.21.4, 
- matplotlib            3.3.3 
- scikit-learn          0.23.2.


## Citation
Please kindly cite the papers [Weighted Feature Fusion of Convolutional Neural Network and Graph Attention Network for Hyperspectral Image Classification](https://ieeexplore.ieee.org/abstract/document/9693311) if this code is useful and helpful for your research.


```
@ARTICLE{9693311,
  author={Dong, Yanni and Liu, Quanwei and Du, Bo and Zhang, Liangpei},
  journal={IEEE Transactions on Image Processing}, 
  title={Weighted Feature Fusion of Convolutional Neural Network and Graph Attention Network for Hyperspectral Image Classification}, 
  year={2022},
  volume={31},
  number={},
  pages={1559-1572},
  doi={10.1109/TIP.2022.3144017}}
```

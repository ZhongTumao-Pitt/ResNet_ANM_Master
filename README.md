
## Development of a plug-and-play anti-noise module for fault diagnosis of rotating machines in nuclear power plants

This libary includes the source code of the paper "Development of a plug-and-play anti-noise module for fault diagnosis of rotating machines in nuclear power plants".

Xianping Zhong, Fei Wang
Feb 15, 2022


## Requirements
- Python 3.7
- Numpy 1.16.2
- Pandas 0.24.2
- Pickle
- tqdm 4.31.1
- sklearn 0.21.3
- Scipy 1.2.1
- opencv-python 4.1.0.25
- PyWavelets 1.0.2
- pytorch >= 1.1
- torchvision >= 0.40


## Datasets
- [CWRU Bearing Dataset](https://csegroups.case.edu/bearingdatacenter/pages/download-data-file/)
- [THU Gearbox Dataset] from the paper "[Wei, Dongdong, et al. "Weighted domain adaptation networks for machinery fault diagnosis." Mechanical Systems and Signal Processing 158 (2021): 107744.]" Proposed by Wei et al.

## Model
- 0 refers to the original ResNet
- 1~9 refer to the 9 variants of ResNet obtained based on the different loading modes of the anti-noise module

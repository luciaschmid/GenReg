# GenReg
Implementation of paper "GenReg: Deep Generative Method for Fast Point Cloud Registration" [[1]](#1)

This implementation is not the official code, it is community code for a university project.

For data as in the paper the datasets ModelNet40 [[2]](#2) and 7Scenes [[3]](#3) are used.

# Used third parties
For the dataloader for the 7Scenes dataset code from the [github implementation](https://github.com/XiaoshuiHuang/fmr)
of the Feature-Metric Registration[[4]](#4) paper is used, as it is referenced in the GenReg paper.

As pytorch does not offer an implementation of the earth mover distance (EMD), we used a third party implementation of [EMD
for pytorch](https://github.com/daerduoCarey/PyTorchEMD) [[5]](#5).

PyTorchEMD is added as a submodule. Please use the following command to get it after cloning this project:
`git submodule update --init --recursive`

## Proposed code of conduct
If you are working on one of the ToDos create a branch with the same or similar name to the todo. When you finished 
the task create a merge request and write in the telegram chat so that a team member reviews your code and completes the merge asap.
Feel free to add missing ToDos.

It would also make sense to stay close to the coding style used in the exercises of the lecture. This way the code will look uniform and tidy.

## ToDos
### Basic Implementation
- edit [main notebook](main.ipynb) (currently only downloads data) to run simple configuration
- [Data ModelNet40](data/ModelNet40.py): complete dataloader and basic data preprocessing (see section 4.1 of paper)
- [Data 7Scenes](data/SevenScenes.py): complete dataloader and basic data preprocessing (section 4.1, be aware that data is in tsdf folder in data after running main notebook)
- implement [PointMixer](model/pointmixer.py) (see section 3.1 and appendix section 1.1)
- implement [Feature Interaction module](model/featureinteraction.py) (see section 3.2)
- implement [decoder](model/decoder.py) (see appendix section 1.1)
- implement [PDSAC](model/pdsac.py) (see section 3.3)
- add forward function for [generator](model/generator.py) (should be done after all parts of generator are implemented to know all necessary parameters)
- implement [discriminator neural network](model/discriminator.py) (see appendix section 1.1)
- complete [training file](training/trainer.py) (see section 3.4 and Appendix section 1.2)
- add visualization file(s) (no designated file/file folder yet)
- run overfit successfully
### Experiments
- first add sub ToDos
- 4.3
- 4.4
- 4.5
- 4.6

## Instructions
To view tensorboard, please use the following command:

`tensorboard --logdir build/log`


Checkpoints are stored in build/checkpoints

## References
<a id="1">[1]</a> 
Xiaoshui Huang, Zongyi Xu, Guofeng Mei, Sheng Li, Jian Zhang, Yifan Zuo, Yucheng Wang.\
GenReg: Deep Generative Method for Fast Point Cloud Registration.\
[ArXiv abs/2111.11783](https://arxiv.org/abs/2111.11783) (2021)

<a id="2">[2]</a>
Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao.\
3D ShapeNets: A Deep Representation for Volumetric Shapes\
Proceedings of 28th IEEE Conference on Computer Vision and Pattern Recognition (CVPR2015)

<a id="3">[3]</a>
Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, Andrew Fitzgibbon.\
Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images\
Proc. Computer Vision and Pattern Recognition (CVPR) | June 2013

<a id="4">[4]</a>
Xiaoshui Huang, Guofeng Mei, Jian Zhang.\
Feature-Metric Registration: A Fast Semi-Supervised Approach for Robust Point Cloud Registration Without Correspondences\
[ArXiv abs/2005.01014](https://arxiv.org/abs/2005.01014) (2020)

<a id="5">[5]</a>
Haoqiang Fan, Hao Su, Leonidas Guibas.\
A Point Set Generation Network for 3D Object Reconstruction from a Single Image\
[ArXiv abs/1612.00603](https://arxiv.org/abs/1612.00603) (2016)
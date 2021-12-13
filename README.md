# GenReg
Implementation of paper "GenReg: Deep Generative Method for Fast Point Cloud Registration" [[1]](#1)

This implementation is not the official code, it is community code for a university project.

## Proposed code of conduct
If you are working on one of the ToDos add your name behind it and create a branch with the same or similar name to the todo. When you finished 
the task create a merge request and write in the telegram chat so that a team member reviews your code and completes the merge asap.
Feel free to add missing ToDos.

It would also make sense to stay close to the coding style used in the exercises of the lecture. This way the code will look uniform and tidy.

## ToDos
### Basic Implementation
- edit [main notebook](main.ipynb) (currently only downloads data) to run simple configuration
- [Data ModelNet40](data/ModelNet40.py): complete dataloader and basic data preprocessing (see section 4.1 of paper)
- [Data 7Scenes](data/7Scenes.py): complete dataloader and basic data preprocessing (section 4.1, be aware that data is in tsdf folder in data after running main notebook)
- implement [PointMixer](model/pointmixer.py) (see section 3.1 and appendix section 1.1)
- implement [Feature Interaction module](model/featureinteraction.py) (see section 3.2)
- implement [decoder](model/decoder.py) (see appendix section 1.1)
- implement [PDSAC](model/pdsac.py) (see section 3.3)
- add forward function for [generator](model/generator.py) (should be done after all parts of generator are implemented to know all necessary parameters)
- implement [discriminator neural network](model/discriminator.py) (see appendix section 1.1)
- complete [training file](training/train_genreg.py) (see section 3.4 and Appendix section 1.2)
- add visualization file(s) (no designated file/file folder yet)
- run overfit successfully
### Experiments
- first add sub ToDos
- 4.3
- 4.4
- 4.5
- 4.6

## References
<a id="1">[1]</a> 
Xiaoshui Huang, Zongyi Xu, Guofeng Mei, Sheng Li, Jian Zhang, Yifan Zuo, Yucheng Wang (2019). 
GenReg: Deep Generative Method for Fast Point Cloud Registration.
ArXiv abs/2111.11783 (2021)

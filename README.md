# nuclei_segmentation

## Requirements

* python 2.7
* numpy
* keras
* tensorflow
* scipy
* cv2
* skimage
* pillow
* h6py
* configParser


## Training and Testing
First run:

```python
python prepare_dataset.py 
```
to prepare dataset

* Training
Directly run:
```python
python run_training.py
```
At the begining of the training, a folder name "test1" will be created in "experiment". During the training, models will be saved in it. 
* Testing
```python
python run_training.py
```
It will generate a folder name '1' in 'test1'. The predictions will be shown in it.
You may modify configuration.txt to change the experiement settings.

## Licence
The code is licensed under [MIT](https://github.com/easycui/nuclei_segmentation/blob/master/LICENSE). Copyright (c) 2018.　The dataset is obtained from https://nucleisegmentationbenchmark.weebly.com/dataset.html.

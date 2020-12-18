# ELEC0134-AMLS-Assignment-2021

This repository provides code and assignment for **ELEC0134 AMLS Assignment 20-21** of **17061069**.

Please note that code from here split the data to train, validation and test set sequentially because the zip file google drive did not divide the dataset to train and test set initially.

### 0. Preparation
#### 1) Required File
Please download the file from [here](https://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/download) 
and place the file named `shape_predictor_68_face_landmarks.dat` to `/Datasets/`.
It is necessary when extracting facial landmarks for pre-processing, especially for `Task A1`.

#### 2) Install requirements
`pip install -r requirement.txt`

It will install required dependencies. However, please install `CUDA` on your own.

#### 1. Run the codes
If you want to run the models that produces the best results,
```
python main.py
``` 
Or, if you want to run the deep learning models manually, go to the folder by
```
cd A1
```
then, 
```
python A1_DL.py
```
for machine learning models,
```
python A1_ML.py
```
You can specify which model to train and which option to give. For example,
```
python A1_DL.py --init 'he' --gap True
```
or,
```
python A1_ML.py --model 'knn'
```
Please refer to the `config.py` for further information.

Then, it will automatically split the dataset, train the data and plot the results on accuracy and loss evaluate using relevant metrics and save heatmap.

If you want to other tasks, change `A1` to `A2`, `B1` or `B2` for specific tasks.


### Development Environment
```
- Windows 10
- NVIDIA GEFORCE RTX 2060
- CUDA 10.0
- Tensorflow-GPU 2.0.0
- Sklearn 0.23.2
- Numpy 1.19.3
- Pandas 1.1.3
- Matplotlib 3.3.2
- dlib 19.21.0
- imutils 0.5.3
```

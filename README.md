[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5976046.svg)](https://doi.org/10.5281/zenodo.5976046)

# DeepPrecip

![alt text](https://github.com/frasertheking/DeepPrecip/blob/main/images/structure.jpg)

DeepPrecip is a deep convolutional multilayer perceptron that takes active radar measurements of the lower atmosphere as input from K-band radar and returns a surface accumulation estimate. DeepPrecip was trained on 8 years of data from nine observation sites across the northern hemisphere. As a general precipitation model, it can estimate both surface rain and snow.

## Installation

<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

If you wish to train and run your own version of DeepPrecip, you can follow the steps below. Note that Python >= 3.7 and Anaconda is required. 

```bash
  git clone https://github.com/frasertheking/DeepPrecip.git
  conda create -n deep-precip --file req.txt
  conda activate deep-precip
  python deep_precip.py
```

## Data

Preprocessed example .CSV datasets are provided in the data folder, however the raw MRR, Pluvio and meteorologic files are also available via the Zenodo link: 
https://doi.org/10.5281/zenodo.5976046

## Train/Test

For optimal performance, we do a 90/10 train/test split on the available observational datasets. However, DeepPrecip was also tested in a leave-one-site-out cross validation and shown to provide good skill at predicting precipitation on completely unseen sites. If you have access to an MRR and would like to test your own data with DeepPrecip, please feel free or reach out to me for assistance. The training study sites and periods are shown below.

![sites](https://github.com/frasertheking/DeepPrecip/blob/main/images/sites.jpg)

## Model Performance

![res1](https://github.com/frasertheking/DeepPrecip/blob/main/images/res1.jpg)
![res2](https://github.com/frasertheking/DeepPrecip/blob/main/images/res2.jpg)


## Support

For support, please email the corresponding author (Fraser King) at fdmking@uwaterloo.ca


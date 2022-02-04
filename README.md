
# DeepPrecip

![alt text](https://github.com/frasertheking/DeepPrecip/blob/main/images/structure.png)

DeepPrecip is a deep convolutional multilayer perceptron that takes active radar measurements of the lower atmosphere as input from K-band radar and returns a surface accumulation estimate. DeepPrecip was trained on 8 years of data from nine observation sites across the northern hemisphere. As a general precipitation model, it can estimate both surface rain and snow.

## Installation

If you wish to train and run your own version of DeepPrecip, you can follow the steps below. Note that Python >= 3.7 and Anaconda is required.

```bash
  git clone https://github.com/frasertheking/DeepPrecip.git
  conda create -n deep-precip --file req.txt
  conda activate deep-precip
  python deep_precip.py
```

## Data

Preprocessed example .CSV datasets are provided in the data folder, however the raw MRR, Pluvio and meteorologic files are also available via the Zenodo link: 


## Train/Test

For optimal performance, we do a 90/10 train/test split on the available observational datasets. However, DeepPrecip was also tested in a leave-one-site-out cross validation and shown to provide good skill at predicting precipitation on completely unseen sites. If you have access to an MRR and would like to test your own data with DeepPrecip, please feel free or reach out to me for assistance. The training study sites and periods are shown below.

![sites](https://github.com/frasertheking/DeepPrecip/blob/main/images/sites.png)

## Model Performance

![res1](https://github.com/frasertheking/DeepPrecip/blob/main/images/res1.png)
![res2](https://github.com/frasertheking/DeepPrecip/blob/main/images/res2.png)


## Support

For support, please email the corresponding author (Fraser King) at fdmking@uwaterloo.ca


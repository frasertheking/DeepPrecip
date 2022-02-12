[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5976046.svg)](https://doi.org/10.5281/zenodo.5976046)

![alt text](https://github.com/frasertheking/DeepPrecip/blob/main/images/logo.png)

# DeepPrecip

DeepPrecip is a deep convolutional multilayer perceptron that takes active radar measurements of the lower atmosphere as input from K-band radar and returns a surface accumulation estimate. DeepPrecip was trained on 8 years of data from nine observation sites across the northern hemisphere. As a general precipitation model, it can estimate both surface rain and snow accumulation at 20-minute temporal resolution.

![alt text](https://github.com/frasertheking/DeepPrecip/blob/main/images/structure.jpg)

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

Attached are precipitation accumulation comparisons between DeepPrecip and a collection of commonly used Z-S and Z-R relationships. For more information on how these were performed, please see our article.

![res1](https://github.com/frasertheking/DeepPrecip/blob/main/images/res1.png)
![res2](https://github.com/frasertheking/DeepPrecip/blob/main/images/res2.png)
![res2](https://github.com/frasertheking/DeepPrecip/blob/main/images/results.png)


## Support

For support, please email the corresponding author (Fraser King) at fdmking@uwaterloo.ca

## License 

Copyright 2022 Fraser King

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

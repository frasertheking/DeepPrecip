![GitHub Workflow Status](https://img.shields.io/github/workflow/status/dwyl/auth_plug/Elixir%20CI?label=build&style=flat-square) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5976046.svg)](https://doi.org/10.5281/zenodo.5976046) 

![alt text](https://github.com/frasertheking/DeepPrecip/blob/main/images/logo.png)

# DeepPrecip

DeepPrecip is a deep convolutional multilayer perceptron that takes active radar measurements of the lower atmosphere as input from K-band radar and returns a surface accumulation estimate. DeepPrecip was trained on 8 years of data from nine observation sites across the northern hemisphere. As a general precipitation model, it can estimate both surface rain and snow accumulation at 20-minute temporal resolution.

![alt text](https://github.com/frasertheking/DeepPrecip/blob/main/images/structure.png)

## Installation

<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

If you wish to train and run your own version of DeepPrecip, you can follow the steps below. Note that Python >= 3.7 and Anaconda is required. 

```bash
  git clone https://github.com/frasertheking/DeepPrecip.git
  conda env create -f req.yml
  conda activate deep_precip
  python deep_precip.py
```

## Data

MRR, Pluvio and meteorologic files are also available via the Zenodo link: 
https://doi.org/10.5281/zenodo.5976046

## Train/Test

For optimal performance, we do a 90/10 train/test split on the available observational datasets. However, DeepPrecip was also tested in a leave-one-site-out cross validation and shown to provide good skill at predicting precipitation on completely unseen sites. If you have access to an MRR and would like to test your own data with DeepPrecip, please feel free or reach out to me for assistance. The training study sites and periods are shown below.

![sites](https://github.com/frasertheking/DeepPrecip/blob/main/images/sites.png)

## Run on IPU

To run DeepPrecip on IPUs, create a new IPU environment by enabling the Poplar SDK & installing the Poplar Tensorflow wheel as described in the IPU [Getting Started Guide](https://docs.graphcore.ai/en/latest/getting-started.html#getting-started), then install the relevant requirements. 

```bash
virtualenv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
source [path_to_SDK]+/poplar-ubuntu_18_04-[poplar_ver]+[build]/enable.sh
pip install tensorflow-[ver]+[platform].whl
pip install -r ipu_requirements.txt
```

Next follow similar instructions as above to clone the repository and download the data:

```
git clone https://github.com/frasertheking/DeepPrecip.git
cd DeepPrecip
mkdir ./runs
mkdir ./checkpoints
wget https://frasertheking.com/downloads/deep_precip_example_data.zip
unzip deep_precip_example_data.zip
```

 Then to run on IPUs, run the following command:

```bash
python deep_precip_ipu.py
```

## Model Performance

Attached are precipitation accumulation comparisons between DeepPrecip and a collection of commonly used Z-S and Z-R relationships. For more information on how these were performed, please see our article.

![res1](https://github.com/frasertheking/DeepPrecip/blob/main/images/res1.png)
![res2](https://github.com/frasertheking/DeepPrecip/blob/main/images/res2.png)

## Throughput
DeepPrecip model training throughput was also examined on a variety of different hardware setups (shown below).

<center><img src="https://github.com/frasertheking/DeepPrecip/blob/main/images/throughput.png" width="450"></center>

## Support

For support, please email the corresponding author (Fraser King) at fdmking@uwaterloo.ca

## License 

Copyright 2022 Fraser King

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

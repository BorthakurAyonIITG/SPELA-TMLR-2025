# SPELA: Learning-Using-a-Single-Forward-Pass
This repository contains the codes used for the paper "Learning-Using-a-Single-Forward-Pass", accepted at TMLR. Note that we provide python notebooks as blueprints of the code used and this can be varied easily for applying to different datasets.

## Getting Started
1. Clone this repository by running the following command:
```bash
git clone https://github.com/BorthakurAyonIITG/SPELA-TMLR-2025.git
```
2. Install the dependencies by running the following command:
```bash
cd SPELA-TMLR-2025
pip install -r requirements.txt
```
3. All the datasets used in this paper are open source. To download the datasets, you can do this by running the following commands:
```bash
cd Data
chmod +x download_datasets.sh
./download_datasets.sh
```
4. The above commands will create a directory for every dataset downloaded and zip the files. Further when running the notebooks, you will need to specify the correct path to the datasets so that `torchvision` can unzip and load the datasets.

## Symmetric Vector Generation
The code to generate the symmetric vectors for every layer is present in the `Models` directory. There are two files, `sphere_points.py` and `sphere_points_CNN.py`, and are used for generating the symmetric vectors for MLPs and CNNs respectively.

## Experiments
We have implemented our algorithm on MLPs and CNNs, further applied it to multiple datasets and performed various ablation studies. Here is a brief overview of the experiments we have conducted:

### MLPs
The implementation of SPELA for MLPs can be found in the `MLP` folder. In our initial experiments with SPELA, we have varied both the number of training epochs and the layer sizes using the notebook `MLP_SPELA_MultiLayer.ipynb`. We have trained this on MNIST-10, KMNIST-10 and FashionMNIST-10 datasets. The results of comparison networks was directly taken from [PEPITA](https://proceedings.mlr.press/v162/dellaferrera22a.html) and [Forward-Forward](https://arxiv.org/abs/2212.13345) papers. 
<!-- > **TODO:** Add file for backprop training and its description here.

### Memory Computations
> **TODO:** Add the files for memory computations and plotting here. -->

### Transfer Learning
Here we used a pre-trained ResNet-50 model to extract features from CIFAR-10, CIFAR-100, Pets-37, Aircraft-100, Food-101 and Flowers-102 datasets. These vectors were then used to train an MLP classifier. To benchmark our algorithm, we trained the MLP classifier on both SPELA and Backpropagation. 

The code for this experiment is in the `TL` folder. The notebooks found in them are as follows:
- `extract_features.ipynb`: This notebook extracts the features from the pre-trained ResNet-50 model.
- `TL_spela.ipynb`: This notebook trains the MLP classifier on the extracted features using SPELA.

### CNNs
We developed a class grouping mechanism for training CNNs using SPELA and showcased its learning capabilities on CIFAR-10, CIFAR-100 and SVHN-10 datasets. The code for this is present in the `CNN` folder, described below:
- `CNN_SPELA_CIFAR_10.ipynb`: This notebook trains the CNN on CIFAR-10 using SPELA.
- `CNN_SPELA_SVHN_10.ipynb`: This notebook trains the CNN on SVHN-10 using SPELA.
- `results_analysis.ipynb`: This notebook is used to analyze the results of the experiments.
- `data/`: This directory is used to store the results for each of the experiments.

Also in our implementation of the `Conv_Layer()` class, we batch the symmetric vectors in an appropriate manner to ensure fast computations of activations and gradients. This can be seen in the `__init__()` function.

## Citation
```Bibtex
@article{somasundaram2025learning,
    title={Learning Using a Single Forward Pass},
    author={Aditya Somasundaram and Pushkal Mishra and Ayon Borthakur},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=EDQ8QOGqjr},
    note={}
}
```

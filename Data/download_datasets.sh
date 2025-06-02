#!/bin/bash

set -e  # Exit on any error

echo "Downloading all datasets"

echo "Downloading CIFAR-10..."
mkdir -p CIFAR-10
cd CIFAR-10
wget -nc https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
cd ..

echo "Downloading CIFAR-100..."
mkdir -p CIFAR-100
cd CIFAR-100
wget -nc https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz
cd ..

echo "Downloading SVHN..."
mkdir -p SVHN-10
cd SVHN-10
wget -nc http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -nc http://ufldl.stanford.edu/housenumbers/test_32x32.mat
cd ..

echo "Downloading Oxford Pets (Pets-37)..."
mkdir -p Pets-37
cd Pets-37
wget -nc https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget -nc https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
cd ..

echo "Downloading FGVC Aircraft (Aircraft-100)..."
mkdir -p Aircraft-100
cd Aircraft-100
wget -nc https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -xzf fgvc-aircraft-2013b.tar.gz
cd ..

echo "Downloading Food-101..."
mkdir -p Food-101
cd Food-101
wget -nc http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz
cd ..

echo "Downloading Flowers-102..."
mkdir -p Flowers-102
cd Flowers-102
wget -nc https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget -nc https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
wget -nc https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
tar -xzf 102flowers.tgz
cd ..

echo "All datasets downloaded and extracted!"
# Feature Engineering for Chip Design

Chips are key components of many applications and tools, from mobile phones to self-driving cars. Chip design involves defining the product requirements for a chip's architecture and system, as well as the physical layout of the chip's individual circuits, a task which is becoming increasingly challenging as these technologies continue to develop.  Due to Moore’s Law, rising complexity is pushing the limits of existing chip design techniques, and machine learning offers a possible avenue to new progress. One specific area where chip designers face problems is congestion; often there are certain areas in a chip through which large amounts of information must pass, creating bottlenecks and reducing efficiency. Although electronic design automation tools have been useful to ensure scalability, reliability and time to market, our group aims to improve the process by exploring self-supervised learning techniques that will create useful features to learn effective graph representations to improve performance in predicting congestion and demand.

Given the relatively unexplored domain of machine learning for chip design, many recent attempts have tried to apply tools from different fields, which may not perfectly fit this specific use-case. For example, chip circuits are often represented as graphs in machine learning, and researchers must choose a specific kind of graph and the features within the graph. Our project aims to identify possible shortcomings in current circuit representations and suggest possible improvements, which will allow for deeper and more accurate research in the future.

Our work is an extension of the work done in the following paper:  
"DE-HNN: An effective neural model for Circuit Netlist representation.
 Z. Luo, T. Hy, P. Tabaghi, D. Koh, M. Defferrard, E. Rezaei, R. Carey, R. Davis, R. Jain and Y. Wang. 27th Intl. Conf. Artificial Intelligence and Statistics (AISTATS), 2024." [arxiv](https://arxiv.org/abs/2404.00477)

## Folder Structure

```
|
└───data
|   └───superblue
└───results
|   |   baselines
|   └───plots
└───src
|   |   models
|   |   pyg_dataset.py
|   |   run_all_data.py
|   |   train_all_cross.py
|   └───visualization.py
└───README.md
└───cuda_related_package.txt
└───requirements.txt
```

## Python Files
(1) ```pyg_dataset.py``` creates a PyTorch object to store the circuit netlist of a chip as well as node and net features, which is then passed into the Graph Neural Network to train the model

(2) ```run_all_data.py``` takes in raw chip data and engineers features that go into the model

(3) ```train_all_cross.py``` initializes the model with the given hyperparameters, then trains for a set number of epochs. It saves the metrics from training as well as the final model file

(4) ```visualization.py``` converts each experiment's metrics into graphs for better visualization and comparison between runs

## Environment Setup

The code is based on Python version 3.8.8 and most of the dependencies can be installed using 
```commandline
pip install -r requirements.txt
```

There are some CUDA-related packages in 
```commandline
cuda_related_package.txt
```
Those packages might need to be manually installed to fit your device's CUDA version. 

If you believe your CUDA version fits the requirements, please run:
```commandline
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Data 

All of the data is available at [link](https://zenodo.org/records/14599896)

Download the file **superblue.zip** first. Then extract and put the folder in the directory "/data" directory. You should be able to see a new directory called **superblue** inside the data folder, which contains all the raw data for the superblue chips.

## How to load the dataset 

After the data is processed downloaded, run the following python file:

```commandline
python run_all_data.py
```

This will do some initial feature engineering so that the chip data is in a format that can be used by the model. 

The new data will be saved at **"superblue/superblue_{design number}/"**

## For running experiments

After dataset are created and saved, next, go to "/src" directory

In the file **train_all_cross.py**, you can set the hyperparameters and config for the model you want to train, and how to split the dataset. Then, run

```commandline
python train_all_cross.py
```

The metrics from training the model will be saved to the directory "/results/baselines" as a csv file. To create plots to compare different model runs, run the following

```commandline
python visualization.py
```

This will save the figures to "/results/plots"

Thank you for running our project!

# Chip Design and Graph Representation

Chip design optimizations have become increasingly challenging due to rising complexity, resulting in recent research towards machine learning methods and other techniques to improve the design process. One such example of a machine learning technique is the DE-HNN model, which uses a directed hyper-graph to improve the outcome of place-and-route tools, such as by modeling the congestion in a given netlist. We build upon the results and architecture of this model to identify specific features in a netlist that are highly related to congestion through exploratory analyses and the implementation of explainable AI tools, such as SHAP (SHapley Additive exPlanations) values. In addition, we propose new approaches to the partitioning process using causal analysis techniques.  We also explore alternatives to the data processing step, such as undersampling and oversampling, to improve the model’s performance on outliers. Finally, we consider reframing the model architecture as a classification problem to improve the identification of congested locations.

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
|   |   |   encoders
|   |   └───layers
|   |   config.json
|   |   pyg_dataset.py
|   |   run_all_data.py
|   |   train_all_cross.py
|   └───visualization.py
└───README.md
└───cuda_related_package.txt
└───requirements.txt
```

## Files in /src
(1) ```/models``` is a folder that contains the classes for the model architecture, which are imported into the training files

(2) ```config.json``` contains the hyperparameters to build and train the model

(3) ```pyg_dataset.py``` creates a PyTorch object to store the circuit netlist of a chip as well as node and net features, which is then passed into the Graph Neural Network to train the model

(4) ```run_all_data.py``` takes in raw chip data and engineers features that go into the model

(5) ```train_all_cross.py``` initializes the model with the given hyperparameters, then trains for a set number of epochs. It saves the metrics from training as well as the final model file

(6) ```visualization.py``` converts each experiment's metrics into graphs for better visualization and comparison between runs

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

## Data Download

All of the data is available at [link](https://zenodo.org/records/14599896)

Download the file **superblue.zip** first. Then extract and put the folder in the directory "/data" directory. You should be able to see a new directory called **superblue** inside the data folder, which contains all the raw data for the superblue chips.

## Data Processing

After the data is downloaded, run the following python file:

```commandline
python run_all_data.py
```

This will do some initial feature engineering so that the chip data is in a format that can be used by the model. 

The new data will be saved at **"superblue/superblue_{design number}/"**

## Running experiments

After dataset are created and saved, next, go to "/src" directory

In the file **"config.json"**, you can set the hyperparameters and config for the model you want to train. Then, run the following file to start the training process.

```commandline
python train_all_cross.py
```

The metrics from training the model will be saved to the directory **"/results/baselines"** as a csv file. To create plots to compare different model runs, go to the file **"visualization.py"** and change the variables shown by the comments at the top to generate the right plot for the data. Then run

```commandline
python visualization.py
```

This will save the figures to **"/results/plots"** where you can view them.

Thank you for running our project!

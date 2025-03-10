# Subpartitioning

This portion of the code is a bit more involved, and requires running some preprocessing scripts. Note that this preprocessing **REQUIRES** a GPU (used to compute eigenvectors for 1000 graphs). If you wish to run the preprocessing scripts, please see the relevant section below. If you do not have access to a GPU, please navigate [here](https://drive.google.com/file/d/1-SP5x2GqUCE0zvnzpEtaFpRK09-1ao9M/view?usp=sharing). Download the ZIP file and extract its contents into this folder. Ultimately, the folder structure should look like:

```
|
└───subpartitioning
|   |   README_SUB.md
|   |   create_eigen_sub.py
|   |   gen_subpartitions.py
|   |   h_dataset_sub.py
|   |   pyg_dataset_sub.py
|   |   run_all_data.py
└───└───train_all_cross.py
```

## Files in /subpartitioning
(1) ```create_eigen_sub.py``` computes the Laplacian Eigenvectors for each subpartition, as well as the degree of each node and net (**REQUIRES** GPU)

(2) ```gen_subpartitions.py``` uses METIS to hierarchically extract partitions from each Superblue chip design

(3) ```pyg_dataset_sub.py``` creates a PyTorch object to store the circuit netlist of each subpartition, as well as node and net features, which is then passed into the Graph Neural Network to train the model

(4) ```run_all_data.py``` aggregates the subpartition data and saves them

(5) ```train_all_cross.py``` initializes the model with the given hyperparameters, then trains for a set number of epochs. It saves the metrics from training, the final model file, and generates loss plots

## Preprocessing

Please note that this step can take up to an hour and **REQUIRES** a GPU. Run:

```commandline
pip install cupy PyMetis==2023.1.1
```

After the additional dependencies are installed, run the following python file:

```commandline
python gen_subpartitions.py
```

and then:

```commandline
python create_eigen_sub.py
```

and then:

```commandline
python run_all_data.py
```

## Training Model

To train the model, run:

```commandline
python train_all_cross.py
```
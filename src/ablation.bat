@echo off
python train_demand.py
python train_demand_noise.py
python train_demand_neighborhood.py
python train_demand_pd.py
python train_demand_eigen.py
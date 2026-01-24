# Yaw Rate and Yaw Estimation

This repository contains the Jupyter notebooks and scripts related to the presented project on **yaw rate and yaw estimation**.

## Project Structure

- **2014 Targa sixty-six/**  
  Contains the telemetry data from *Revs* used in the project.

- **MS-NN.ipynb**  
  Notebook containing the implementation of the **Model-Structured Neural Network (MS-NN)**.

- **NN.ipynb**  
  Notebook containing the implementation of the **black-box Neural Network**.

- **Fig/**  
  Contains the figures and results obtained from the experiments.

## How to Run the Project

1. Run the Python script `build_dataset.py`:
   - Download of telemetries
   - Data preprocessing  
   - Downsampling  
   - Dataset creation  

2. Execute the Jupyter notebook of interest (from either `MS-NN` or `NN`).

3. Check the results and plots in the `Fig` folder.

## Notes

- Make sure all required Python dependencies are installed before running the scripts and notebooks.
- The workflow is designed to first prepare the datasets and then run the selected neural network models.

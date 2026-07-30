# Few-Shot Parameter-Efficient Fine-Tuning and Continual Active Learning

This is a codebase for the Few-Shot Parameter-Efficient Fine-Tuning and Continual Active Learning (FS-PEFT-CAL) project.

## Running the full training pipeline

The full training pipeline for this project can be performed by executing the code in the file `training/training.py`. The sequential steps necessary for the execution are described in this file in detail. 

### Data preparation

Before beginning the training process, the data needs to be prepared by creating the training and test files for each dataset. Please refer to the directory `data/` and the file `training/training.py` for details.

### Hyperparameter tuning

The next step is conducting hyperparameter tuning. In this case, hyperparameter tuning involves two steps. The first step consists of tuning the hyperparameters for CL methods for each dataset while keeping the PEFT module settings fixed. The second step involves fixing the tuned hyperparameters for the CL methods obtained in the first step and tuning the hyperparameters for the PEFT methods for each dataset. After each step, the results need to be evaluated. Please refer to the directories `utility/create_tuning/` and `evaluation/`, and the file `training/training.py` for details.

### Training

Once the optimal configurations for CL and PEFT have been tuned, training with CAL can be conducted. Two types of baselines can also be assessed: the AL baseline and the full dataset training baseline. After these results are obtained, they can be evaluated, and the necessary plots and tables can be generated. Please refer to the directories `utility/create_training/`, `utility/create_baselines/` and `evaluation/`, and the file `training/training.py` for details.

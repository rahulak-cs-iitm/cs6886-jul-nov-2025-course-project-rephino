# REPhINO (CS6886 Project)

**REPhINO** (REliable Physics-informed Intelligence through Neural Operators) is a project aimed at developing reliable, fault-tolerant Neural operators suitable for critical edge applications. The project was completed as part of the CS6886 Systems Engineering for Deep Learning course at IIT Madras (July-Nov 2025 Semester).

## Contributors

- Pradyumnan R
- Rahul Anilkumar

## Project Organization

The project is organized into two main sections:

1. Sloshing Neural Operator (Pradyumnan)
2. NeuralOps_EmotionEngine (Rahul)

## 1. Sloshing Neural Operator

### Problem Setup

The problem undersconsideration is that of sloshing of a fuel inside an axisymmetric tank. The problem setup is inspired from [González et al.](https://www.researchgate.net/profile/Alvaro-Romero-Calvo/publication/377851094_Open-Source_Propellant_Sloshing_Modeling_and_Simulation/links/65e0444fe7670d36abe61763/Open-Source-Propellant-Sloshing-Modeling-and-Simulation.pdf). The paper gives a MATLAB solver that will give the eigenfunctions for the following parameters:

- Fuel physical properties
- Fuel fill percentage
- Axisymmetric tank shape
- Gravitational acceleration

### Modifications

However, the solver described above doesn't provide a way to obtain the final velocity fields that are of interest to calculate forces and torques on the fuel tank. Instead it provides mechanical equivalents. Obtaining the velocity fields and sloshing surface shape requires complicated mathematical reverse engineering, which I setup using my own modifications to the solver. These take the order of a few minutes to run to completion, which makes it unsuitable for edge applications. I used this modified solver to generate the dataset to train the Sloshing Neural Operator.

### Contribution

The tools available today (November, 2025) aren't capable of instantaneous inferencing of velocity fields given a forcing function. Using Fourier Neural Operators [Li et al.](https://doi.org/10.48550/arXiv.2010.08895), I have designed and trained a model that can inference the velocity fields in an autoregressive manner for a cylindrical tank containing water and under a constant gravitational field, experiencing a time varying lateral acceleration. Further improvements can be made by training the model on more diverse data.


### Files

The [sloshing_neural_operator](/sloshing_neural_operator/) directory has several files. The important ones and their purpose are given below. Instructions to use them are given in the next section.

- `neuraloperator`: This is the submodule containing the neural operators designed by _Li et al._
- `best_model.pt`: This is the checkpoint of the best model, containing the model state along with the optimizer and LR schedular states to continue training if desired.
- `evaluate_4_channel.ipynb`: This is the notebook that allows for easy evaluation of the model performance, along with somme visualizations.
- `mat_to_torch_tensor.py`: This is the file that allows for converting `.mat` data files obtained from MATLAB to `.pt` files that store torch tensors.
- `sloshing_neural_op_training_4_channels.py`: This is the training script for training a neural operator from scratch.
- `test_paths.txt`: This contains the various various test files. Required by `evaluate_4_channel.ipynb`.

## 2. NeuralOps_EmotionEngine

This section contains the code for Reliability Analysis of the FNO, through Fault injection campaigns. It also contains the code to apply selective TMR and Activation Clipping to the FNO on the basis of the fault simulation results.

### Setup

Setup scripts for setting up the required packages. To install necessary packages use:
```
pip install -r requirements.txt
```

### Baseline

This folder contains the initial simulation runs using a baseline FNO for Darcy flow, experimenting with single and multi-bit fault injections on activations to see the impact on output. To run baseline:
```
make baseline
```

### Run

Run Folder contains the Fault injection runs and Fault tolerance code for Baseline Darcy FNO and Sloshing FNO. Sloshing FNO folder contains LayerWiseFaultInjection and FaultTolerance using the Slosh FNO model. The make commands are the same.

#### LayerWiseFaultInjection

Contains code for Isolated_SEU, Clustered_MBU/Concentrated_MBU and Clustered_MBU/Distributed_MBU arranged as separate folders.

To run, use make commands in the respective folders:
```
make isolatedSEU
```
```
make concentratedMBU
```
```
make distributedMBU
```

#### FaultTolerance

This folder contains code for Selective TMR (TMRFaultTolerance) and Activation Clipping (Activation Clipping). To run the scripts, use the commands:

**TMR Analysis** — Get number of parameters, identify which ones to be strengthened and potential memory savings compared to full system TMR:

```
make tmr_analysis
```

**TMR Implementation** — Implement the selective TMR using register buffers and Majority voting logic:

```
make tmr_implementation
```

**Collect Max Activation** — Estimate the maximum activation values per layer (required for Activation Clipping):
```
make collectMaxActivation
```

**Activation Clipping** — Run the Activation Clipping code:
```
make activationClipping
```

### Output

Plots and logs from the original runs are saved in the respective folders.

### Data for Sloshing FNO Runs

Data and Checkpoints for Sloshing FNO Runs are uploaded [here](https://drive.google.com/drive/folders/1Zw2MBFpyYQfSYnrTh8N3xwRG0jqd3XU6?usp=drive_link) due to space constraints.




# REPhINO (CS6886 Project)

**REPhINO** (REliable Physics-informed Intelligence through Neural Operators) is a project aimed at developing reliable, fault-tolerant Neural operators suitable for critical edge applications. The project was completed as part of the CS6886 Systems Engineering for Deep Learning course at IIT Madras (July-Nov 2025 Semester).

## Contributors

- Pradyumnan R
- Rahul Anilkumar

## Project Organization

The project is organized into two main sections:

1. Sloshing Neural Operator (Pradyumnan)
2. NeuralOps_EmotionEngine (Rahul)

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




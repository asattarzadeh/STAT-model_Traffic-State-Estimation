![image](https://github.com/asattarzadeh/STAT-model_Traffic-State-Estimation/assets/92281234/4f00d1cd-89c6-47cf-81f0-f9026f138033)# Traffic State Estimation

This repository contains the code for the traffic state estimation model described in the paper "Traffic State Estimation with Spatio-Temporal Autoencoding Transformer (STAT Model)".

## Files

- `data/`: Contains the data file `pems-bay.h5`.
- `src/`: Contains the source code for the data loader, model, training, and utility functions.
- `notebooks/`: Contains the Jupyter Notebook for traffic state estimation.
- `results/`: Contains example results.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Traffic-State-Estimation.git
   cd Traffic-State-Estimation

## Model Architecture

The architecture of the Spatio-Temporal Autoencoding Transformer (STAT) Model is depicted below. The model is designed to effectively capture both spatial and temporal dependencies in traffic state estimation.

![Model Architecture](images/model_architecture.png)

### Explanation

The STAT model combines various components to enhance its predictive capabilities:

- **GRU**: Gated Recurrent Units (GRU) are used to handle temporal dependencies efficiently.
- **Dense Layer**: These layers are used for feature transformation and capturing non-linear relationships.
- **1D-Convolution**: 1D-Convolutional layers help in extracting local temporal patterns from the traffic data.
- **Transformer Block**: The transformer encoder and decoder layers are used to model long-range dependencies and complex interactions between features.
- **Multi-Head Attention**: This mechanism allows the model to focus on different parts of the input sequence, enhancing its ability to capture spatial-temporal dependencies.
- **Positional Encoding**: Positional encodings are added to the input embeddings to provide information about the position of each element in the sequence.

The combination of these components allows the STAT model to achieve superior performance in traffic state estimation tasks by effectively modeling both global and local interactions.

## Model Results

The following tables compare the performance of the STAT model with other baseline algorithms for traffic state estimation on different datasets.

![Performance Comparison](results/results_comparison.png)

### Explanation

The results are evaluated using three metrics: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Square Error (RMSE) across different prediction horizons (15 min, 30 min, and 60 min). The tables show that the STAT model consistently outperforms other models across all metrics and horizons, demonstrating its superior capability in capturing both spatial and temporal dependencies in traffic data.

- **Table I**: Performance comparison for the I980 dataset.
- **Table II**: Performance comparison for the PEMS-BAY dataset

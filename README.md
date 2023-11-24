# RMPEM
Codes for “Random Mask Perturbation Based Explainable Method of Graph Neural Networks”

## Running the Program
To execute the program, use the following command in your terminal:

```python main.py -cuda=0 -gm=GNN -data=ba2 -retrain=0```

### Command Explanation
- -cuda=0: This option sets the CUDA device ID to 0, typically used for GPU acceleration.
- -gm=GNN: fixed now, black-box model can be extended.
- -data=ba2: Five datasets are optional, ba2, nci-h23, ptc-fr, tox21-ar, mutag.
- -retrain=0: This flag indicates whether the model should be retrained or not, with '0' meaning no retraining.

### Configuring Explanation Method in config.yml
The choice of explanation method can be configured in the config.yml file. This allows for flexible control over which method is employed by the program.


### System Requirements
- Python installed (preferably Python 3.7).
- Relevant libraries and dependencies for the program are installed.
  - torch 1.8.0
  - torch-geometric 2.2.0
- A compatible CUDA-enabled GPU if using CUDA features (optional).

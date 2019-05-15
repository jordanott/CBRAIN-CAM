# Parallel Hyper Parameter Optimization for Conservation Models

The `runner.py` script handles creating jobs with the SGE Scheduler. The number of concurrent jobs can be set using the `--max_concurrent` flag.  
The `main.py` script is run by each individual process to create a network and train it. The results are then reported back to the runner via `client.keras_send_metrics` in `model.py`.

## Available Configurations

* Conservation Network
	* MSE Loss
	* Weak Loss

* Normal Network
	* MSE Loss
	* Weak Loss

```
python3.6 runner.py --net_type [normal or conservation] --loss_type [mse or weak_loss]
```

The results directory mirrors these configurations:  
```
SherpaResults/{data}/{net_type}_{loss_type}/  
├── Models  
└── output  
```

Where `Models` contains the saved architecure and weights in `h5` files and `output` contains training information provided via Sherpa.


## Model Diagnostics
Model Diagnostics can easily be run via:  
`python3.6 process.py --net_type [normal or conservation] --loss_type [mse or weak_loss] --diagnostics`

This script iterates through the models in `SherpaResults/{data}/{net_type}_{loss_type}/Models/`:  
1. Loads the model
2. Runs diagnostics via `ModelDiagnostics(NN,config_fn,data_fn)`
3. Outputs the results to `SherpaResults/{data}/{net_type}_{loss_type}/Diagnostics/` with the file name matching the model it corresponds to
4. Outputs plots of loss and val_loss to `SherpaResults/{data}/{net_type}_{loss_type}/Diagnostics/Plots/`

# Hyper Parameter Optimization for Conservation Models

### To Do
- [ ] Move to parallel trials (currently issues with SGE Scheduler)
- [ ] Move to labeled tensors for conservation network
- [ ] Optimize variable choices in conservation equation

### Currently Running
* Baseline model
  * No conservation loss
  * No conservation network
  * `python main.py --run_type baseline`
* Optimizing parameters of network with conservation loss
  * `python main.py --run_type hyper_param_opt`
* Optimizing parameters of conservation network with conservation loss
  * `python main.py --run_type hyper_param_opt_conservation`


  Running a local search from this configuration  
  ```
  default_params = {
    num_layers:       5,
    nodes per layer:  512,
    alpha:            .1,
    loss:             weak_loss,
    dropout:          0.25,
    batch_norm:       True,
    leaky_relu:       .3,
    lr:               0.001,
  }
  ```

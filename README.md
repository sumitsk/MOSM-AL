# Active Learning with Multi-Output Spectral Mixture Gaussian Processes

## Installation
This project is aimed at actively learning the distribution of multiple correlated target features (e.g. concentration of minerals in a region). The distribution of target features or variables are modelled by a Multi-Output Spectral Mixture Gaussian Process proposed in https://papers.nips.cc/paper/7245-spectral-mixture-kernels-for-multi-output-gaussian-processes (https://github.com/gparracl/MOSM). After estimating the model hyperparameters by fitting the model on a training set, the agent actively determines the type and location of samples to be collected in order to minimize the uncertainty of model's prediction. The information gain criterion is entropy. 

### Requirements: 
* [GPFlow 1.0](https://github.com/GPflow/GPflow)

After installing the listed dependencies, simply clone this package to run scripts.

## Getting Started
See `ex.py` script to setup the Jura dataset, train the GP model and perform active learning. You can setup certain arguments from command line (see `arguments.py` file). Execute the following to start the training: 
```
python ex.py
```
A significant portion of the code is taken from Gabriel Parra's implementation available at https://github.com/gparracl/MOSM.

## Contact
For any queries, feel free to raise an issue or contact me at sumitsk@cmu.edu.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

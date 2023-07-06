# Anti-Symmetric DGN
This repository provides the official reference implementation of our papers 

- **[Anti-Symmetric DGN: a stable architecture for Deep Graph Networks](https://openreview.net/forum?id=J3Y7cgZOOS)** accepted at ICLR 2023
- **[Non-Dissipative Propagation by Anti-Symmetric Deep Graph Networks](https://drive.google.com/file/d/1uPHhjwSa3g_hRvHwx6UnbMLgGN_cAqMu/view?usp=share_link)** accepted at the DLG-AAAI’23 workshop.

Please consider citing us

	@inproceedings{gravina2023adgn,
		author = {Alessio Gravina and Davide Bacciu and Claudio Gallicchio},
	 	title = {{Anti-Symmetric DGN: a stable architecture for Deep Graph Networks}},
	 	booktitle = {The Eleventh International Conference on Learning Representations },
	 	year = {2023},
		url = {https://openreview.net/forum?id=J3Y7cgZOOS}
	}

## Requirements
_Note: we assume Miniconda/Anaconda is installed, otherwise see this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for correct installation. The proper Python version is installed during the first step of the following procedure._

1. Install the required packages and create the environment
    - ``` conda env create -f env.yml ```

2. Activate the environment
    - ``` conda activate adgn ```

## Run the experiment
_Note: To run the experiment is fundamental to define the model name and dataset name into the run_all.sh files. For more details launch ```python3 main.py --help``` or ```python3 run-adgn-2-8.py --help```_
- Graph property prediction

    ```graph_prop_pred/run_all.sh```

- Graph benchmarks

    ```graph_benchmark/run_all.sh```
    - _Note: please ensure that conf_seeds_5.json and conf_seeds_5.json match the generated seed before running the experiment_

- Graph heterophilic benchmarks

    ```graph_heteropily/run_all.sh```

- Tree-NeighborsMatch

    ```Tree-NeighborsMatch/run_all.sh```

## Repository structure
The repository is structured as follows:

    ├── README.md                <- The top-level README.
    │
    ├── env.yml                  <- The conda environment requirements.
    │
    ├── conda_list_output.txt    <- The specific version of each package in the adgn environment.
    │
    ├── graph_benchmark          <- Contains the code to reproduce the graph benchmarks experiment.
    │    ├── conf_seeds_5.json   <- The configuration seeds used in the experiment.
    │    ├── data_seeds_5.json   <- The seeds used in the experiment to split the dataset into train/valid/test.
    │    ├── run_all.sh          <- The script used to run the experiment. Note: you need to specify the name of the model and the dataset (see conf.py and utils/__init__.py)
    │    ├── models              <- Contains the code for the framework A-DGN and other DGNs 
    │    ├── utils               <- Contains the code for data and io utilities.
    │    ├── main.py             <- The main.
    │    ├── train.py            <- Implements the code responsible for training and evaluation of the models.
    │    ├── conf.py             <- Contains the hyper-parameter space for each model.
    │    └── model_selction.py   <- Implements the model selection and produces the report of the results.
    │
    └── graph_prop_pred          <- Contains the code to reproduce the graph property prediction experiment.
    │   ├── run_all.sh           <- The script used to run the experiment. Note: you need to specify the name of the model and the dataset (see conf.py and utils/__init__.py)
    │   ├── models               <- Contains the code for the framework A-DGN and other DGNs 
    │   ├── utils                <- Contains the code for data and io utilities.
    │   ├── main.py              <- The main.
    │   ├── train_GraphProp.py   <- Implements the code responsible for training and evaluation of the models.
    │   ├── conf.py              <- Contains the hyper-parameter space for each model.
    │   └── model_selction.py    <- Implements the model selection and produces the report of the results.
    │    
    ├── graph_heteropily          <- Contains the code to reproduce the graph heterophilic benchmarks experiment.
    │    ├── run_all.sh          <- The script used to run the experiment. Note: you need to specify the name of the model and the dataset (see conf.py and utils/__init__.py)
    │    ├── models              <- Contains the code for the framework A-DGN and other DGNs 
    │    ├── utils               <- Contains the code for data and io utilities.
    │    ├── main.py             <- The main.
    │    ├── train.py            <- Implements the code responsible for training and evaluation of the models.
    │    ├── conf.py             <- Contains the hyper-parameter space for each model.
    │    └── model_selction.py   <- Implements the model selection and produces the report of the results.
    │
    └── Tree-NeighborsMatch      <- Contains the code to reproduce the Tree-NeighborsMatch experiment retrieved from the original source code: https://github.com/tech-srl/bottleneck.
        ├── run_all.sh           <- The script used to run the experiment. Note: you need to specify the name of the model and the dataset (see conf.py and utils/__init__.py)
        └── run-adgn-2-8.py      <- The main.

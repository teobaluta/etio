# ETIO
### Membership Inference Attacks and Generalization: A Causal Perspective

## Setting up your development environment
  
**1. Install an environment manager**  
It is recommended to use an environment management tool like [virtualenv](https://virtualenv.pypa.io/en/stable/) to easily manage the project's requirements and avoid any conflicts with your system packages. If you are using virtualenv, run these commands to set up a development environment:
```sh
$ cd into/your/project/folder
$ virtualenv -p python3 env
$ source env/bin/activate
```
This will activate an empty virtual environment. You can use ```deactivate``` to exit the environment.

At this point, your project folder will have an `env` directory for your virtual environment, and a `code` directory for your code.

**2. Install dependencies**  
In the root folder (where the requirements.txt file is), run this command to install all dependencies:
```sh
$ pip install -r requirements.txt
```
### Requirements
The main requirements are as follows:
```
Python 3.8.10
```
Python Packages
```
torch                   1.7.1+cu110
torchvision             0.8.2+cu110
tqdm                    4.61.2
matplotlib              3.4.2
scipy                   1.7.1
```

Command to install Cuda:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm matplotlib
```

# Functions
1. Preparing datasets: please run `./estimator/prepare_dataset.py`
2. Training models: please run `./trainer/train.py`
3. Computing bias & variance: plaese run `./estimator/estimator.py`

More details about each step can be found in `README.md` in each folder (e.g., `estimator` and `trainer`).


**3. Performing Attacks**
Attacks can be performed with the scripts in the `attack` module. The details are in `README.md` file in `attack` folder.

**4. Collecting Statistics**
We store the trained models and attack models in a folder with following structure:
```
    ├──────────────────────────────────── 
    │base_dir
    │── datasets
    │── mlleaks-attacks
    │   ├── mlleaks-attacks-wd_5e3                                  
    │   │   ├── epoch_400                   
    │   │   └── epoch_500                           
    │   └── mlleaks-attacks-wd_5e4
    │   │   ├── epoch_400        
    │   │   └── epoch_500                   
    ├── mlleaks_top3-attacks                    
    │   ├── mlleaks_top3-attacks-wd_5e3                 
    │   │   ├── epoch_400
    │   │   ├── epoch_500 
    │   └── mlleaks_top3-attacks-wd_5e4 
    │       ├── epoch_400 
    │       └── epoch_500
    ├── oak17-attacks/
    │   ├── oak17-attacks-wd_5e3
    │   │   ├── epoch_400
    │   │   └── epoch_500
    │   └── oak17-attacks-wd_5e4
    │       ├── bad-stuff
    │       ├── epoch_400
    │       └── epoch_500
    ├───────────────────────────────────
```

To generate the summary of all trained models, along with the attacks performed over them, run the following script:

```bash
python3 analyze_attacks/parse_summary.py <base_dir> <summary_dir>
```




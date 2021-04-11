### LIBRARIES ###
# Global libraries
import yaml

# Custom libraries
from run_exp import run_experiment

### MAIN CODE ###
# Import the global configuration
with open("cfg.yml", "r") as yml_file:
    cfg = yaml.safe_load(yml_file)

# For each list parameter, run different experiments
run_experiment(cfg)
### LIBRARIES ###
# Global libraries
import os
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint

# Custom libraries
from models.TransE.run_model import run_transe
from models.DistMult.run_model import run_distmult
from models.ComplEx.run_model import run_complex
from models.RotatE.run_model import run_rotate
from models.pRotatE.run_model import run_protate
from models.ConvE.run_model import run_conve
from models.rgcn.run_model import run_rgcn

### FUNCTION DEFINITION ###
def run_experiment(cfg):
    """Runs a single experiment.

    Args:
        cfg: Dict
            configuration to use
    """
    if cfg["use_wandb"]:
        wandb.init(project="mlns_kg_embedding")
        wandb.config.dataset = cfg["dataset"]
        wandb.config.model_name = cfg["model_name"]
        wandb.config.n_epochs = cfg["n_epochs"]

        wandb.config.batch_size = cfg["batch_size"]
        wandb.config.negative_sample_size = cfg["negative_sample_size"]

    # Load the dataset
    dataset_path = os.path.join("data", "datasets", cfg["dataset"])

    # Load the pretrained model
    ckpt_name = f"{cfg['model_name']}-{cfg['dataset']}-{cfg['n_epochs']}"
    ckpt_file = os.path.join("checkpoints", f"{ckpt_name}.ckpt")

    if not os.path.exists(ckpt_file):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_score",
            dirpath="checkpoints",
            filename=ckpt_name,
            save_top_k=1,
            mode="min",
        )
    else:
        checkpoint_callback = None

    if cfg["model_name"] == "TransE":
        run_transe(dataset_path, ckpt_file, checkpoint_callback, cfg)
    elif cfg["model_name"] == "DistMult":
        run_distmult(dataset_path, ckpt_file, checkpoint_callback, cfg)
    elif cfg["model_name"] == "ComplEx":
        run_complex(dataset_path, ckpt_file, checkpoint_callback, cfg)
    elif cfg["model_name"] == "RotatE":
        run_rotate(dataset_path, ckpt_file, checkpoint_callback, cfg)
    elif cfg["model_name"] == "pRotatE":
        run_protate(dataset_path, ckpt_file, checkpoint_callback, cfg)
    elif cfg["model_name"] == "ConvE":
        run_conve(dataset_path, ckpt_file, checkpoint_callback, cfg)
    elif cfg["model_name"] == "R-GCN":
        run_rgcn(dataset_path, ckpt_file, cfg)
    else:
        raise ValueError("Wrong model name given")

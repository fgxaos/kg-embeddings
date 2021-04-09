### LIBRARIES ###
# Global libraries
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Custom libraries
from data.datasets.dataloader import KGDataModule

from models.TransE.TransE import TransE
from models.DistMult.DistMult import DistMult
from models.ComplEx.ComplEx import ComplEx
from models.RotatE.RotatE import RotatE
from models.RotatE.pRotate import pRotatE

### MAIN CODE ###
def run_experiment(dataset: str, model_name: str):
    """Runs a single experiment.

    Args:
        dataset: str
            name of the dataset to use
        model_name: str
            name of the model to use
    """
    # Load the dataset
    dataset_path = os.path.join("data", "datasets", dataset)
    data_module = KGDataModule(dataset_path, 2, 4, 128)

    # Create a model instance
    n_entity = len(data_module.entity2id)
    n_relation = len(data_module.relation2id)
    if model_name == "TransE":
        model = TransE(n_entity, n_relation, 256, 12.0, 1e-3)
    elif model_name == "DistMult":
        model = DistMult(n_entity, n_relation, 256, 12.0, 1e-3)
    elif model_name == "ComplEx":
        model = ComplEx(n_entity, n_relation, 256, 12.0, 1e-3)
    elif model_name == "RotatE":
        model = RotatE(n_entity, n_relation, 256, 12.0, 1e-3)
    elif model_name == "pRotatE":
        model = pRotatE(n_entity, n_relation, 256, 12.0, 1e-3)
    else:
        raise ValueError("Wrong model name given")

    # Load the pretrained model
    ckpt_path = os.path.join("checkpoints", f"{model_name}-{dataset}")
    if not os.path.exists(ckpt_path):
        # Train the model
        checkpoint_callback = ModelCheckpoint(
            monitor="val_score",
            dirpath="checkpoints",
            filename="{model_name}-{dataset}-{epoch}",
            save_top_k=1,
            mode="min",
        )
        trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
        trainer.fit(model, datamodule=data_module)

    # Test the pretrained model
    if model_name == "TransE":
        model_test = TransE.load_from_checkpoint(ckpt_path)
    elif model_name == "DistMult":
        model_test = DistMult.load_from_checkpoint(ckpt_path)
    elif model_name == "ComplEx":
        model_test = ComplEx.load_from_checkpoint(ckpt_path)
    elif model_name == "RotatE":
        model_test = RotatE.load_from_checkpoint(ckpt_path)
    elif model_name == "pRotatE":
        model_test = pRotatE.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError("Wrong model name given")
    model_test.test()


run_experiment("wn18rr", "TransE")

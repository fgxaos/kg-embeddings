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
from models.ConvE.ConvE import ConvE

### VARIABLES ###
n_epochs = 10

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
    data_module = KGDataModule(
        dataset_path, num_workers=4, batch_size=128, negative_sample_size=128
    )

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
    elif model_name == "ConvE":
        model = ConvE(n_entity, n_relation, 20, 10, 32, 3, 0.2, 0.2, 0.3, 0.0, 1e-3)
    else:
        raise ValueError("Wrong model name given")

    # Load the pretrained model
    ckpt_name = f"{model_name}-{dataset}-{n_epochs}"
    ckpt_file = os.path.join("checkpoints", f"{ckpt_name}.ckpt")
    if not os.path.exists(ckpt_file):
        # Train the model
        checkpoint_callback = ModelCheckpoint(
            monitor="val_score",
            dirpath="checkpoints",
            filename=ckpt_name,
            save_top_k=1,
            mode="min",
        )
        trainer = pl.Trainer(max_epochs=n_epochs, callbacks=[checkpoint_callback])
        trainer.fit(model, datamodule=data_module)

    # Test the pretrained model
    if model_name == "TransE":
        model_test = TransE.load_from_checkpoint(ckpt_file)
    elif model_name == "DistMult":
        model_test = DistMult.load_from_checkpoint(ckpt_file)
    elif model_name == "ComplEx":
        model_test = ComplEx.load_from_checkpoint(ckpt_file)
    elif model_name == "RotatE":
        model_test = RotatE.load_from_checkpoint(ckpt_file)
    elif model_name == "pRotatE":
        model_test = pRotatE.load_from_checkpoint(ckpt_file)
    else:
        raise ValueError("Wrong model name given")

    trainer = pl.Trainer()
    trainer.test(model_test, datamodule=data_module)


run_experiment("wn18rr", "TransE")

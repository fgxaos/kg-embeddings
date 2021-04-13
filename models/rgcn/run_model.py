### LIBRARIES ###
# Global libraries
import os

import pytorch_lightning as pl

# Custom libraries
from models.rgcn.dataloader import KGDataModule
from models.rgcn.rgcn import RGCN

### FUNCTION DEFINITION ###
def run_rgcn(
    dataset_path: str,
    ckpt_file: str,
    checkpoint_callback,
    cfg: dict,
):
    """Runs experiment with the R-GCN model.

    Args:
        dataset_path: str
            path to the dataset
        ckpt_file: str
            checkpoint file to the pretrained model
        checkpoint_callback: ModelCheckpoint or None
            callback to use to save the checkpoint
        cfg: dict
            configuration dictionary to use
    """
    cfg_model = cfg["rgcn"]["model"]
    cfg_data = cfg["rgcn"]["data"]
    cfg_training = cfg["rgcn"]["training"]

    # Load the dataset
    data_module = KGDataModule(
        dataset_path,
        num_workers=cfg_data["num_workers"],
        batch_size=cfg_data["batch_size"],
        negative_sample_size=cfg_data["negative_sample_size"],
    )

    # Create a model instance
    n_entity = len(data_module.entity2id)
    n_relation = len(data_module.relation2id)
    model = RGCN(
        n_entity,
        n_relation,
        data_module.all_triples,
        cfg_model["n_bases"],
        cfg_model["dropout"],
        cfg_training["learning_rate"],
        cfg_model["reg_ratio"],
    )

    # Load the pretrained model
    if not os.path.exists(ckpt_file):
        # Train the model
        trainer = pl.Trainer(
            max_epochs=cfg_training["n_epochs"],
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, datamodule=data_module)

    # Test the pretrained model
    model_test = RGCN.load_from_checkpoint(ckpt_file)
    trainer = pl.Trainer()
    trainer.test(model_test, datamodule=data_module)
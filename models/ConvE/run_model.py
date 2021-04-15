### LIBRARIES ###
# Global libraries
import os

import pytorch_lightning as pl

# Custom libraries
from models.ConvE.dataloader import KGDataModule
from models.ConvE.ConvE import ConvE

### FUNCTION DEFINITION ###
def run_conve(
    dataset_path: str,
    ckpt_file: str,
    checkpoint_callback,
    cfg: dict,
):
    """Runs experiment with the ConvE model.

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
    cfg_model = cfg["ConvE"]["model"]
    cfg_data = cfg["ConvE"]["data"]
    cfg_training = cfg["ConvE"]["training"]

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
    model = ConvE(
        n_entity, n_relation, 20, 10, 32, 3, 0.2, 0.2, 0.3, 0.0, 1e-3, False, 0.1
    )
    model = ConvE(
        n_entity,
        n_relation,
        cfg_model["embedding_size_h"],
        cfg_model["embedding_size_w"],
        cfg_model["conv_channels"],
        cfg_model["conv_kernel_size"],
        cfg_model["embedding_dropout"],
        cfg_model["feature_map_dropout"],
        cfg_model["proj_layer_dropout"],
        cfg_model["regularization"],
        cfg_training["learning_rate"],
        cfg["use_wandb"],
        cfg_model["label_smooth"],
    )

    # Load the pretrained model
    # if not os.path.exists(ckpt_file):
    if checkpoint_callback:
        # Train the model
        trainer = pl.Trainer(
            max_epochs=cfg_training["n_epochs"], callbacks=[checkpoint_callback]
        )
        trainer.fit(model, datamodule=data_module)

    # Test the pretrained model
    model_test = ConvE.load_from_checkpoint(ckpt_file)
    trainer = pl.Trainer()
    trainer.test(model_test, datamodule=data_module)
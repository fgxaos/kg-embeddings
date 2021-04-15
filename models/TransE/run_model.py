### LIBRARIES ###
# Global libraries
import os

import pytorch_lightning as pl

# Custom libraries
from models.TransE.dataloader import KGDataModule
from models.TransE.TransE import TransE

### FUNCTION DEFINITION ###
def run_transe(
    dataset_path: str,
    ckpt_file: str,
    checkpoint_callback,
    cfg: dict,
):
    """Runs experiment with the TransE model.

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
    cfg_model = cfg["TransE"]["model"]
    cfg_data = cfg["TransE"]["data"]
    cfg_training = cfg["TransE"]["training"]

    # Load the dataset
    data_module = KGDataModule(
        dataset_path,
        num_workers=cfg_data["num_workers"],
        train_batch_size=cfg_data["train_batch_size"],
        val_batch_size=cfg_data["val_batch_size"],
        negative_sample_size=cfg_data["negative_sample_size"],
    )

    # Create a model instance
    n_entity = len(data_module.entity2id)
    n_relation = len(data_module.relation2id)
    model = TransE(
        n_entity,
        n_relation,
        cfg_model["hidden_dim"],
        cfg_model["gamma"],
        cfg_training["learning_rate"],
        cfg["use_wandb"],
        cfg_model["double_entity_embedding"],
        cfg_model["double_relation_embedding"],
        cfg_model["negative_adversarial_learning"],
        cfg_model["adversarial_temperature"],
        cfg_model["uni_weight"],
        cfg_model["regularization"],
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
    model_test = TransE.load_from_checkpoint(ckpt_file)
    trainer = pl.Trainer()
    trainer.test(model_test, datamodule=data_module)
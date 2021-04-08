### LIBRARIES ###
# Global libraries
import os
import pytorch_lightning as pl

# Custom libraries
from data.datasets.dataloader import KGDataModule
from models.TransE.TransE import TransE

### MAIN CODE ###
dataset = "FB15k-237"
# dataset = "wn18rr"
# Load the FB15k-237 dataset
data_module = KGDataModule(os.path.join("data", "datasets", dataset), 2, 4, 128)
# Create a model
n_entity = len(data_module.entity2id)
n_relation = len(data_module.relation2id)
model = TransE(n_entity, n_relation, 256, 12.0, 1e-3)
# Train the model
trainer = pl.Trainer()
trainer.fit(model, datamodule=data_module)

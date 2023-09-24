import numpy as np
import nibabel as nib
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchir.utils import IRDataSet
from torchir.metrics import NCC
from torchir.dlir_framework import DLIRFramework
from torchir.networks import DIRNet
from torchir.transformers import BsplineTransformer

DEST_DIR = Path('./output')

class CTDataSet(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load NIFTI image
        image = nib.load(self.image_paths[idx]).get_fdata()
        
        # Convert the image data to float32 type, as PyTorch's default is float32
        image = np.asarray(image, dtype=np.float32)

        # Add channel dimension: [1, depth, height, width]
        image = np.expand_dims(image, axis=0)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image

# Define the path to the directory containing NIFTI images
image_dir = Path("E:/s4692034/thorax_resampled")
image_paths = [str(p) for p in image_dir.glob("*.nii")]

transform = None

ct_dataset = CTDataSet(image_paths=image_paths, transform=transform)
registration_dataset = IRDataSet(ct_dataset)

val_set_size = 10
test_set_size = 80
train_set_size = len(ct_dataset) - val_set_size - test_set_size

ds_train_subset, ds_validation_subset, ds_test_subset = random_split(ct_dataset, 
    [train_set_size, val_set_size, test_set_size], generator=torch.Generator().manual_seed(617))

ds_train = IRDataSet(ds_train_subset, num_pairs=30)
ds_validation = IRDataSet(ds_validation_subset, num_pairs=15)

batch_size = 1
training_batches = 25
validation_batches = 10

torch.manual_seed(617)
train_sampler = torch.utils.data.RandomSampler(ds_train, replacement=True, 
                                               num_samples=training_batches*batch_size)

train_loader = torch.utils.data.DataLoader(ds_train, batch_size, sampler=train_sampler, num_workers=0)
val_loader = torch.utils.data.DataLoader(ds_validation, batch_size, num_workers=0)


class LitDLIRFramework(pl.LightningModule):
    def __init__(self, only_last_trainable=True):
        super().__init__()
        self.dlir_framework = DLIRFramework(only_last_trainable=only_last_trainable)
        self.add_stage = self.dlir_framework.add_stage
        self.metric = NCC()
    
    def configure_optimizers(self):
        lr = 0.002
        weight_decay = 0
        optimizer = torch.optim.Adam(self.dlir_framework.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        return {'optimizer': optimizer}

    def forward(self, fixed, moving):
        warped = self.dlir_framework(fixed, moving)
        return warped
    
    def training_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/training', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/validation', loss)
        return loss
    
    def on_epoch_end(self):
        torch.cuda.empty_cache()


def main():
    model = LitDLIRFramework()
    torch.set_float32_matmul_precision('medium')
    model.add_stage(network=DIRNet(grid_spacing=(16, 16, 16), kernels=8, num_conv_layers=5, num_dense_layers=2, ndim=3),
                    transformer=BsplineTransformer(ndim=3, upsampling_factors=(16, 16, 16)))
    trainer = pl.Trainer(default_root_dir=DEST_DIR,
                         log_every_n_steps=50,
                         val_check_interval=1.0,
                         max_epochs=50,
                         devices=1, 
                         accelerator="gpu")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
    
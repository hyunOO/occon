from argparse import ArgumentParser
import math

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from pl_bolts.metrics import precision_at_k


class SupervisedModel(pl.LightningModule):
    def __init__(
        self,
        arch: str = 'resnet18',
        image_size: int = 224,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = self.load_backbone(arch, image_size, kwargs['num_classes'])
        #self.projector = deepcopy(self.encoder.fc)
        #self.encoder.fc = nn.Identity()
        #self.projector = nn.Identity()

    def load_backbone(self, arch, image_size, num_classes):
        assert image_size in [32, 224]
        backbone = models.__dict__[arch](zero_init_residual=True, num_classes=num_classes)
        if image_size == 32:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
        backbone.feat_dim = backbone.fc.weight.shape[1]
        self.reset_parameters(backbone)
        return backbone

    def reset_parameters(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            if isinstance(m, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -bound, bound)

    def init_lr_schedule(self):
        base_lr = self.hparams.base_lr * self.hparams.global_batch_size / 256  # linear lr scaling rule
        final_lr = self.hparams.final_lr
        max_epochs = self.hparams.max_epochs

        lr_schedule = torch.tensor([
            final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t / max_epochs))
            for t in torch.arange(max_epochs)
        ])
        return lr_schedule

    def configure_optimizers(self):
        lr = self.hparams.base_lr * self.hparams.global_batch_size / 256  # linear lr scaling rule
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs[0]
        #predicted = self.encoder(inputs)
        #logits = self.projector(predicted)
        logits = self.encoder(inputs)

        loss = F.cross_entropy(logits, labels)
        acc1, acc5 = precision_at_k(logits, labels, top_k=(1, 5))

        self.log_dict({'loss': loss, 'acc1': acc1, 'acc5': acc5}, prog_bar=True)
        return loss

    def validation_step(self, *args, **kwargs):
        pass  # no validation error

    def forward(self, x):
        #return self.projector(self.encoder(x))
        return self.encoder(x)
        #return self.encoder(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--arch", default="resnet18", type=str)
        parser.add_argument("--num_classes", default=1000, type=int)
        parser.add_argument("--image_size", default=224, type=int)

        # training params
        parser.add_argument("--base_lr", default=0.03, type=float)
        parser.add_argument("--final_lr", default=1e-3, type=float)
        parser.add_argument('--global_batch_size', default=None, type=int)  # default: inference mode
        parser.add_argument("--min_crop_scale", default=0.08, type=float)  # stronger augmentation
        parser.add_argument("--max_crop_scale", default=1., type=float)
        parser.add_argument("--jitter_strength", default=0, type=float)
        parser.add_argument("--gaussian_blur", default=False, type=bool)

        return parser

import torch.nn as nn
import torch.optim as optim
from src import data_loader
from src import config
from src.model import CIFARNet, Trial
from torch.optim.lr_scheduler import OneCycleLR


def main(norm_type="bn"):
    net = CIFARNet(
        first_layer_output_size=config.FIRST_LAYER_OUTPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        norm_type=norm_type,
    ).to(config.DEVICE)
    criterion = nn.functional.nll_loss
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = OneCycleLR(optimizer, max_lr=0.5, total_steps=20)
    train_loader, test_loader = data_loader.get_iterators()

    run = Trial(
        name=norm_type,
        model=net,
        args={
            "epochs": config.EPOCH,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "optimizer": optimizer,
            "loss_fn": criterion,
            "scheduler": scheduler,
        },
    )

    run.run()
    print("Done!")

    return run

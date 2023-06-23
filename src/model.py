from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src import config, norm

tqdm = partial(tqdm, position=0, leave=True)


# generates conv blocks with the following two configs :
# Conv->Relu->BN->Dropout | DepthWiseSeparableConv->Relu->BN->Dropout
def conv_block(
    num_layers,
    in_channel,
    padding,
    norm_func,
    norm_type,
    dialation=1,
    depthwise=False,
    kernel_size=3,
    dropout=config.DROPOUT_RATE,
):
    module_list = []
    for j in range(num_layers):
        if not depthwise:
            module_list.extend(
                [
                    nn.Conv2d(
                        in_channel,
                        in_channel * 2,
                        kernel_size=kernel_size,
                        dilation=dialation,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    norm_func(num_channels=in_channel * 2)
                    if norm_type != "bn"
                    else norm_func(num_features=in_channel * 2),
                ]
            )
        else:
            module_list.extend(
                [
                    DepthWiseSeparableConv(
                        in_channel,
                        in_channel * 2,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    norm_func(num_channels=in_channel * 2)
                    if norm_type != "bn"
                    else norm_func(num_features=in_channel * 2),
                ]
            )
        if j != num_layers - 1:
            module_list.append(nn.Dropout(dropout))
        in_channel = in_channel * 2

    return nn.Sequential(*module_list), in_channel


# Transition block contains MaxPooling + 1x1 Conv2d kernel to reduce the
#  number of channels
def transition_block(
    in_channel, out_channel, norm_func, norm_type, other_layers=True
):
    if out_channel > in_channel:
        raise ValueError(
            f"out_channels {out_channel} should be lower than in_channels"
            f" {in_channel} for the 1x1 convolution"
        )
    output = []
    # output.extend(
    #     [
    #         nn.MaxPool2d(2, 2),
    #         nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), bias=False),
    #     ]
    # )
    output.extend(
        [
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), bias=False),
            nn.MaxPool2d(2, 2),
        ]
    )

    if other_layers:
        output.extend(
            [
                nn.ReLU(),
                norm_func(num_channels=out_channel)
                if norm_type != "bn"
                else norm_func(num_features=out_channel),
                nn.Dropout(config.DROPOUT_RATE),
            ]
        )

    return nn.Sequential(*output)


# DepthwiseSeparable Convolution is a grouped Convolution operation followed
#  by a 1x1 convolution
class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.depthwise_layer = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=padding,
            groups=int(out_channels / in_channels),
        )
        self.separable_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

    def forward(self, x):
        return self.separable_layer(self.depthwise_layer(x))


def get_normalization_layer(norm_type):
    factories = {
        "bn": norm.BatchNorm,
        "gn": norm.GroupNorm,
        "ln": norm.LayerNorm,
    }
    try:
        return factories[norm_type]().get_norm()
    except KeyError:
        raise (
            """Normalization Layer not found, \
            please select one of bn, gn, ln"""
        )


class CIFARNet(nn.Module):
    def __init__(self, first_layer_output_size, num_classes, norm_type="bn"):
        super(CIFARNet, self).__init__()
        self.num_classes = num_classes
        self.norm = get_normalization_layer(norm_type)
        self.first_block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=first_layer_output_size,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            self.norm(num_channels=first_layer_output_size)
            if norm_type != "bn"
            else self.norm(num_features=first_layer_output_size),
            nn.Dropout(config.DROPOUT_RATE),
        )
        self.conv_block1, final_out_conv1 = conv_block(
            num_layers=1,
            in_channel=first_layer_output_size,
            kernel_size=3,
            padding=1,
            depthwise=False,
            norm_func=self.norm,
            norm_type=norm_type,
        )
        self.trans_block1 = transition_block(
            in_channel=final_out_conv1,
            out_channel=first_layer_output_size,
            norm_type=norm_type,
            norm_func=self.norm,
        )

        self.conv_block2, final_out_conv2 = conv_block(
            num_layers=3,
            in_channel=first_layer_output_size,
            kernel_size=3,
            padding=1,
            dialation=2,
            norm_type=norm_type,
            norm_func=self.norm,
            depthwise=False,
        )

        self.trans_block2 = transition_block(
            in_channel=final_out_conv2,
            out_channel=first_layer_output_size,
            norm_type=norm_type,
            norm_func=self.norm,
        )

        self.conv_block3, final_out_conv_3 = conv_block(
            num_layers=3,
            in_channel=first_layer_output_size,
            kernel_size=3,
            padding=1,
            depthwise=True,
            norm_type=norm_type,
            norm_func=self.norm,
        )

        # self.trans_block3 = transition_block(
        #     in_channel=final_out_conv_3,
        #     out_channel=num_classes,
        #     other_layers=False,
        # )
        self.GAP = nn.AvgPool2d(5)
        self.conv_block4 = nn.Conv2d(
            in_channels=final_out_conv_3,
            out_channels=config.NUM_CLASSES,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.first_block(
            x
        )  # RF = 3, output_shape : [batch_size, 3, 32, 32]
        x = self.trans_block1(
            self.conv_block1(x)
        )  # RF = 8, output_shape : [batch_size, 32, 16, 16]
        x = self.trans_block2(
            self.conv_block2(x)
        )  # RF = 26 , output_shape : [batch_size, 32, 6, 6]
        x = self.conv_block3(x)
        # RF = 50, output_shape : [batch_size, 10, 3, 3]
        x = self.conv_block4(self.GAP(x)).view(-1, self.num_classes)
        return F.log_softmax(x, dim=1)


class Record:
    def __init__(self, train_acc, train_loss, test_acc, test_loss):
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.test_acc = test_acc
        self.test_loss = test_loss


class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_acc = []
        self.train_loss = []
        self.test_acc = []
        self.test_loss = []

    def train(
        self,
        epochs,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        scheduler=None,
    ):
        for epoch in range(epochs):
            print(f"{epoch + 1} / {epochs}")

            self._train(train_loader, optimizer, loss_fn)
            self._evaluate(test_loader)
            if scheduler:
                scheduler.step()

        return Record(
            self.train_acc, self.train_loss, self.test_acc, self.test_loss
        )

    def _train(self, train_loader, optimizer, loss_fn):
        self.model.train()
        correct = 0
        train_loss = 0

        for _, (data, target) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()

            output = self.model(data)
            loss = loss_fn(output, target)

            train_loss += loss.detach()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        self.train_loss.append(train_loss * 1.0 / len(train_loader.dataset))
        self.train_acc.append(100.0 * correct / len(train_loader.dataset))

        print(
            f" Training loss = {train_loss * 1.0 / len(train_loader.dataset)},"
            " Training Accuracy :"
            f" {100.0 * correct / len(train_loader.dataset)}"
        )

    def _evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for _, (data, target) in tqdm(
                enumerate(test_loader), total=len(test_loader)
            ):
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                output = self.model(data)
                test_loss += torch.nn.functional.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset) * 1.0
        self.test_loss.append(test_loss)
        self.test_acc.append(100.0 * correct / len(test_loader.dataset))

        print(
            f" Test loss = {test_loss}, Test Accuracy :"
            f" {100.0 * correct / len(test_loader.dataset)}"
        )


class Trial:
    def __init__(self, name, model, args):
        self.name = name
        self.model = model
        self.args = args
        self.Record = Record
        self.Trainer = Trainer(model)

    def run(self):
        self.Record = self.Trainer.train(**self.args)

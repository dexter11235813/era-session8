from functools import partial

import matplotlib.pyplot as plt
import torch
from numpy import ceil, clip, floor, sqrt, transpose
from tqdm import tqdm
from numpy import max

# from data_loader import get_iterators
from src import config

tqdm = partial(tqdm, leave=True, position=0)


def get_images(train_loader, num_images=10):
    images, labels = next(iter(train_loader))
    fig = plt.figure(figsize=(15, 8))

    for i in range(num_images):
        img = images[i].squeeze(0).numpy()
        ax = fig.add_subplot(2, 5, i + 1)
        img = img / 2 + 0.5
        img = transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.set_title(str(config.CLASSES[labels[i].item()]))


# Code from here-on borrowed is from the submission made for assignment 6
def get_images_by_classification(
    model, test_loader, device, misclassified=True
):
    mis = []
    mis_pred = []
    mis_target = []

    model.eval()
    with torch.no_grad():
        for _, (data, target) in tqdm(
            enumerate(test_loader), total=len(test_loader)
        ):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            # https://discuss.pytorch.org/t/viewing-misclassified-image-predictions/82410
            if misclassified:
                idx = pred.eq(target.view_as(pred)) == False
            else:
                idx = pred.eq(target.view_as(pred)) == True

            misclassified_target = target.view_as(pred)[idx]
            missclassified_pred = pred[idx]
            missclassified = data[idx]

            mis_pred.append(missclassified_pred)
            mis_target.append(misclassified_target)

            mis.append(missclassified)

    mis = torch.cat(mis)
    mis_pred = torch.cat(mis_pred)
    mis_target = torch.cat(mis_target)

    return mis, mis_pred, mis_target


def plot_images_by_classification(
    number,
    model,
    test_loader,
    device,
    misclassified=True,
    save_path="./images",
    normalize_type="bn",
    n_rows=None,
    n_cols=None,
):
    images, predicted, actual = get_images_by_classification(
        model, test_loader, device, misclassified
    )
    nrows = int(floor(sqrt(number))) if not n_rows else n_rows
    ncols = int(ceil(sqrt(number))) if not n_cols else n_cols
    if misclassified:
        save_path = f"{save_path}/misclassified_images_{normalize_type}.png"
    else:
        save_path = (
            f"{save_path}/correctly_classified_images_{normalize_type}.png"
        )
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j

            ax[i, j].set_title(
                f"Predicted: {config.CLASSES[predicted[index]]},\nActual :"
                f" {config.CLASSES[actual[index]]}"
            )
            ax[i, j].axis("off")
            ax[i, j].imshow(
                clip(transpose(images[index].cpu().numpy(), (1, 2, 0)), 0, 1)
            )

    fig.savefig(save_path, bbox_inches="tight")
    print(f"plot saved at {save_path}")


def plot_curves_for_trials(*trials):
    train_loss = [
        [tl.cpu() for tl in trial.Record.train_loss] for trial in trials
    ]
    train_acc = [trial.Record.train_acc for trial in trials]
    test_loss = [trial.Record.test_loss for trial in trials]
    test_acc = [trial.Record.test_acc for trial in trials]

    data_acc = [train_acc, test_acc]
    data_loss = [train_loss, test_loss]

    legends = [trial.name for trial in trials]
    titles_loss = ["Train loss", "Test loss"]
    titles_acc = ["Train accuracy", "Test accuracy"]

    fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))

    print(f"Max Training Accuracy: {max(train_acc)}")
    print(f"Max Validation Accuracy: {max(test_acc)}")

    for i in range(2):
        ax1[i].set_title(titles_loss[i])

        for k, legend in enumerate(legends):
            ax1[i].plot(data_loss[i][k], label=legend)

        ax1[i].legend()

    for j in range(2):
        ax2[j].set_title(titles_acc[j])

        for k, legend in enumerate(legends):
            ax2[j].plot(data_acc[j][k], label=legend)

        ax2[j].legend()

    fig1.savefig(config.SUMMARY_LOSS, bbox_inches="tight")
    fig2.savefig(config.SUMMARY_ACC, bbox_inches="tight")

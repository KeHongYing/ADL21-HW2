import matplotlib.pyplot as plt
import sys
import json
import numpy as np


def plot(train, val, mode, img_name):
    plt.plot(
        list(range(len(train))),
        train,
        color="orange",
        label="train",
    )
    plt.plot(list(range(len(val))), val, color="red", label="val")
    plt.title("learning curve")
    plt.xlabel("epoch")
    plt.ylabel(mode)
    plt.legend()
    plt.savefig(img_name)
    plt.clf()


with open(sys.argv[1]) as f:
    data = json.load(f)


train_len = 17576
val_len = 4393
batch_size = 12

for mode in ["EM", "F1"]:
    step = int(np.ceil(train_len / batch_size))
    train_curve = [
        np.mean(data["train"][mode][i : i + step])
        for i in range(0, len(data["train"][mode]), 1832)
    ]
    print(len(data["train"][mode]) / 1832)
    step = int(np.ceil(val_len / batch_size))
    val_curve = [
        np.mean(data["val"][mode][i : i + step])
        for i in range(0, len(data["val"][mode]), 204)
    ]
    print(len(data["val"][mode]) / 204)
    plot(train_curve, val_curve, mode, f"{sys.argv[2]}_{mode}.png")

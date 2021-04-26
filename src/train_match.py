import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer

from dataset import MatchDataset
from model import MatchClassifier

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    total_correct, total_loss = 0, 0
    current = 0
    cnt = 0

    model.train()
    for batch, data in enumerate(dataloader):
        correct = 0
        cnt += 1

        token = data["token"].to(device)
        labels = data["label"].to(device)

        pred = model(token)["label"]
        loss = loss_fn(pred, labels)

        correct += (
            ((pred > 0.5).type(torch.float) == labels).type(torch.float).sum().item()
        )
        total_correct += correct
        total_loss += loss
        correct /= pred.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current += len(token)
        loss = loss.item()
        print(
            f"Acc: {(100 * correct):>4.1f}%, loss: {loss:>7f}, [{current:>6d}/{size:>6d}]",
            end="\r",
        )

    total_correct /= current
    total_loss /= cnt
    print(
        f"Acc: {(100 * total_correct):>4.1f}%, loss: {total_loss:>7f}, [{current:>6d}/{size:>6d}]",
    )


def test(dataloader, model, loss_fn, device):
    size = 0

    total_correct, total_loss = 0, 0
    cnt = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            token = data["token"].to(device)
            labels = data["label"].to(device)

            cnt += 1

            pred = model(token)["label"]
            total_loss += loss_fn(pred, labels).item()
            total_correct += (
                ((pred > 0.5).type(torch.float) == labels)
                .type(torch.float)
                .sum()
                .item()
            )
            size += pred.shape[0]

    total_loss /= cnt
    total_correct /= size

    print(f"Val Acc: {(100 * total_correct):>4.1f}%, Val loss: {total_loss:>7f}")

    return total_correct, total_loss


def main(args):
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    data_paths = {split: args.cache_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, MatchDataset] = {
        split: MatchDataset(split_data, tokenizer) for split, split_data in data.items()
    }

    dataloader = {
        split: DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=datasets[split].train_collate_fn
            if split == TRAIN
            else datasets[split].val_collate_fn,
        )
        for split in SPLITS
    }

    torch.manual_seed(args.seed)

    model = MatchClassifier(model_name="hfl/chinese-roberta-wwm-ext").to(args.device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-7, patience=5
    )

    loss_fn = torch.torch.nn.BCELoss()
    max_acc, min_loss = 0, 100
    early_stop = 0
    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch + 1}")
        train(dataloader[TRAIN], model, loss_fn, optimizer, args.device)
        acc, loss = test(dataloader[DEV], model, loss_fn, args.device)

        scheduler.step(loss)

        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), args.ckpt_dir / f"{args.model}_best.pt")
            print(f"model is better than before, save model to {args.model}_best.pt")

        if loss > min_loss:
            early_stop += 1
        else:
            early_stop = 0
            min_loss = loss

        if early_stop == 10:
            print("Early stop...")
            break

    print(f"Done! Best model Acc: {(100 * max_acc):>4.1f}%")
    torch.save(model.state_dict(), args.ckpt_dir / f"{args.model}.pt")

    with open("result_intent.txt", "a") as f:
        f.write(f"{args.model}, {max_acc:>5f}\n")

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/match/",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="model name.",
        default="model",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=500)

    # misc
    parser.add_argument("--seed", type=int, default=0xB06902074)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer

from dataset import MatchDataset
from model import MatchClassifier

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def iter_loop(dataloader, model, loss_fn, optimizer, device, mode):
    total_correct = 0
    total_loss = 0
    current = 0
    cnt = 0

    if mode == TRAIN:
        model.train()
    elif mode == DEV:
        model.eval()

    with torch.set_grad_enabled(mode == TRAIN):
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"[{mode:>5}]")
                cnt += 1

                token = data["token"].to(device)
                label = data["label"].to(device)

                pred = model(token)["label"]

                loss = loss_fn(pred, label)
                correct = ((pred > 0.5).type(torch.long) == label).sum().item()
                total_correct += correct
                total_loss += loss
                correct /= pred.shape[0]

                if mode == TRAIN:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                current += token.shape[0]
                tepoch.set_postfix(loss=f"{loss.item():>.4f}", Acc=f"{correct:>.4f}")

            total_correct /= current
            total_loss /= cnt
            print(
                "\033[2K\033[2K"
                + f"[{mode:>5}] "
                + f" Acc: {total_correct:>.2f},"
                + f" loss: {total_loss:>.7f},"
            )

    return total_correct, total_loss


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.backbone)
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
            collate_fn=datasets[split].collate_fn
            if split == TRAIN
            else datasets[split].predict_collate_fn,
        )
        for split in SPLITS
    }

    torch.manual_seed(args.seed)

    model = MatchClassifier(model_name=args.backbone).to(args.device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-7, patience=5
    )

    loss_fn = torch.torch.nn.BCELoss()
    max_acc, min_loss = 0, 100
    early_stop = 0
    backbone = (
        args.backbone if "/" not in args.backbone else args.backbone.split("/")[1]
    )
    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch + 1}")
        iter_loop(dataloader[TRAIN], model, loss_fn, optimizer, args.device, TRAIN)
        acc, loss = iter_loop(
            dataloader[DEV], model, loss_fn, optimizer, args.device, DEV
        )

        scheduler.step(loss)

        if acc > max_acc:
            max_acc = acc
            torch.save(
                model.state_dict(),
                args.ckpt_dir / f"{args.model}_{backbone}_best.pt",
            )
            print(
                f"model is better than before, save model to {args.model}_{backbone}_best.pt"
            )

        if loss > min_loss:
            early_stop += 1
        else:
            early_stop = 0
            min_loss = loss

        if early_stop == 10:
            print("Early stop...")
            break

    print(f"Done! Best model Acc: {(100 * max_acc):>.4f}%")
    torch.save(model.state_dict(), args.ckpt_dir / f"{args.model}_{backbone}.pt")

    with open("result_match.txt", "a") as f:
        f.write(f"{args.model}_{backbone}, {max_acc:>5f}\n")


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
        default="./cache/match",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/match",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="model name.",
        default="model",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=500)

    # misc
    parser.add_argument("--seed", type=int, default=0xB06902074)

    # model
    parser.add_argument(
        "--backbone",
        help="bert backbone",
        type=str,
        default="voidful/albert_chinese_large",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

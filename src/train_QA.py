import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer

from dataset import QADataset
from model import QAClassifier

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def iter_loop(dataloader, model, loss_fn, optimizer, device, mode):
    size = 2 * len(dataloader.dataset)
    total_start_correct, total_end_correct = 0, 0
    total_start_sentence_correct, total_end_sentence_correct = 0, 0
    total_start_loss, total_end_loss = 0, 0
    current = 0
    cnt = 0

    if mode == TRAIN:
        model.train()
    elif mode == DEV:
        model.eval()

    for batch, data in enumerate(dataloader):
        start_correct, end_correct = 0, 0
        start_sentence_correct, end_sentence_correct = 0, 0
        cnt += 1

        token = data["token"].to(device)
        start = data["start"].to(device)
        end = data["end"].to(device)
        index = data["index"].to(device)

        pred = model(token)

        start_loss = loss_fn(pred["start"].transpose(1, 2), start)
        end_loss = loss_fn(pred["end"].transpose(1, 2), end)

        count = 0
        for idx, (head, tail) in enumerate(index):
            start_correct += (
                (pred["start"][idx][head:tail].argmax(dim=-1) == start[idx][head:tail])
                .type(torch.float)
                .sum()
                .item()
            )
            end_correct += (
                (pred["end"][idx][head:tail].argmax(dim=-1) == end[idx][head:tail])
                .type(torch.float)
                .sum()
                .item()
            )
            start_sentence_correct += (
                torch.all(
                    pred["start"][idx][head:tail].argmax(dim=-1)
                    == start[idx][head:tail],
                    dim=-1,
                )
                .sum()
                .item()
            )
            end_sentence_correct += (
                torch.all(
                    pred["end"][idx][head:tail].argmax(dim=-1) == end[idx][head:tail],
                    dim=-1,
                )
                .sum()
                .item()
            )
            count += tail - head
        start_correct /= count
        end_correct /= count
        # start_correct = (
        #     (pred["start"].argmax(dim=-1) == start).type(torch.float).sum().item()
        # ) / start.shape[1]
        # end_correct = (pred["end"].argmax(dim=-1) == end).type(
        #     torch.float
        # ).sum().item() / end.shape[1]

        # start_sentence_correct = (
        #     torch.all(pred["start"].argmax(dim=-1) == start, dim=-1)
        #     .type(torch.float)
        #     .sum()
        #     .item()
        # )
        # end_sentence_correct = (
        #     torch.all(pred["start"].argmax(dim=-1) == start, dim=-1)
        #     .type(torch.float)
        #     .sum()
        #     .item()
        # )

        total_start_correct += start_correct
        total_end_correct += end_correct
        total_start_sentence_correct += start_sentence_correct
        total_end_sentence_correct += end_sentence_correct
        total_start_loss += start_loss
        total_end_loss += end_loss

        # start_correct /= pred["start"].shape[0]
        # end_correct /= pred["end"].shape[0]

        start_sentence_correct /= pred["start"].shape[0]
        end_sentence_correct /= pred["end"].shape[0]

        if mode == TRAIN:
            optimizer.zero_grad()
            (start_loss + end_loss).backward()
            optimizer.step()

        current += token.shape[0]
        print(
            "\033[2K\033[2K"
            + f"[{mode:>5}] "
            + f"start Acc: {(100 * start_correct):>4.1f}%,"
            + f" end Acc: {(100 * end_correct):>4.1f}%,"
            + f" start sent. Acc: {(100 * start_sentence_correct):>4.1f}%,"
            + f" end sent. Acc: {(100 * end_sentence_correct):>4.1f}%,"
            + f" start loss: {start_loss:>7f},"
            + f" end loss: {end_loss:>7f},"
            + f" [{current:>6d}/{size:>6d}]",
            end="\r",
        )

    total_start_correct /= current
    total_end_correct /= current
    total_start_sentence_correct /= current
    total_end_sentence_correct /= current
    total_start_loss /= cnt
    total_end_loss /= cnt
    print(
        "\033[2K\033[2K"
        + f"[{mode:>5}] "
        + f"start Acc: {(100 * total_start_correct):>4.1f}%,"
        + f" end Acc: {(100 * total_end_correct):>4.1f}%,"
        + f" start sent. Acc: {(100 * total_start_sentence_correct):>4.1f}%,"
        + f" end sent. Acc: {(100 * total_end_sentence_correct):>4.1f}%,"
        + f" start loss: {total_start_loss:>7f},"
        + f" end loss: {total_end_loss:>7f},"
        + f" [{current:>6d}/{size:>6d}]",
        end="\r",
    )

    return (
        total_start_sentence_correct + total_end_sentence_correct
    ) / 2, total_start_loss + total_end_loss


def main(args):
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    data_paths = {split: args.cache_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, QADataset] = {
        split: QADataset(split_data, tokenizer) for split, split_data in data.items()
    }

    dataloader = {
        split: DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=datasets[split].collate_fn,
        )
        for split in SPLITS
    }

    torch.manual_seed(args.seed)

    model = QAClassifier(model_name="hfl/chinese-roberta-wwm-ext").to(args.device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-7, patience=5
    )

    loss_fn = torch.torch.nn.CrossEntropyLoss()
    max_acc, min_loss = 0, 100
    early_stop = 0
    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch + 1}")
        iter_loop(dataloader[TRAIN], model, loss_fn, optimizer, args.device, TRAIN)
        acc, loss = iter_loop(
            dataloader[DEV], model, loss_fn, optimizer, args.device, DEV
        )

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

    with open("result.txt", "a") as f:
        f.write(f"{args.model}, {max_acc:>5f}\n")


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
        default="./ckpt/",
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

import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import BertTokenizer

from dataset import QADataset
from model import QAClassifier

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]
learning_curve = {TRAIN: {"EM": [], "F1": []}, DEV: {"EM": [], "F1": []}}


def iter_loop(dataloader, model, loss_fn, optimizer, device, mode):
    total_sentence_correct = 0
    total_start_loss, total_end_loss = 0, 0
    total_F1 = 0

    if mode == TRAIN:
        model.train()
    elif mode == DEV:
        model.eval()

    with torch.set_grad_enabled(mode == TRAIN):
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"[{mode:>5}]")

                token = data["token"].to(device)
                start = data["start"].to(device)
                end = data["end"].to(device)

                pred = model(token)["start_end"]

                start_loss = loss_fn(pred[..., 0], start)
                end_loss = loss_fn(pred[..., 1], end)
                pred_start = pred[..., 0].argmax(dim=-1)
                pred_end = pred[..., 1].argmax(dim=-1)

                sentence_correct = (
                    ((pred_start == start) & (pred_end == end))
                    .type(torch.float)
                    .mean()
                    .item()
                )

                zero = torch.tensor(0).to(device)
                tp = torch.maximum(
                    torch.minimum(pred_end, end) - torch.maximum(pred_start, start) + 1,
                    zero,
                )
                precision = tp / (torch.maximum(pred_end - pred_start + 1, zero) + 1e-9)
                recall = tp / (torch.maximum(end - start + 1, zero) + 1e-9)
                F1 = (
                    (2 * precision * recall / (precision + recall + 1e-9)).mean().item()
                )

                learning_curve[mode]["EM"].append(sentence_correct)
                learning_curve[mode]["F1"].append(F1)

                total_sentence_correct += sentence_correct
                total_start_loss += start_loss
                total_end_loss += end_loss
                total_F1 += F1

                if mode == TRAIN:
                    optimizer.zero_grad()
                    (start_loss + end_loss).backward()
                    optimizer.step()

                tepoch.set_postfix(
                    s_loss=f"{start_loss.item():>.4f}",
                    e_loss=f"{end_loss.item():>.4f}",
                    Acc=f"{sentence_correct:>.4f}",
                    F1=f"{F1:>.4F}",
                )

            total_sentence_correct /= len(tepoch)
            total_start_loss /= len(tepoch)
            total_end_loss /= len(tepoch)
            total_F1 /= len(tepoch)
            print(
                "\033[2K\033[2K"
                + f"[{mode:>5}] "
                + f" Acc: {total_sentence_correct:>.4f},"
                + f" F1: {total_F1:>.4f},"
                + f" start loss: {total_start_loss:>.4f},"
                + f" end loss: {total_end_loss:>.4f},",
            )

    return total_sentence_correct, total_start_loss + total_end_loss


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.backbone)
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

    model = QAClassifier(model_name=args.backbone, no_pretrained=args.no_pretrained).to(
        args.device
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-7, patience=5
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    max_acc, min_loss = 0, 100
    early_stop = 0
    backbone = Path(
        args.backbone if "/" not in args.backbone else args.backbone.split("/")[1]
    )

    ckpt_dir = args.ckpt_dir / backbone
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / Path(f"{args.model}_config.pkl"), "wb") as f:
        pickle.dump(model.backbone.config, f)

    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch + 1}")
        for split in SPLITS:
            acc, loss = iter_loop(
                dataloader[split], model, loss_fn, optimizer, args.device, split
            )

        scheduler.step(loss)

        if acc > max_acc:
            max_acc = acc
            torch.save(
                model.state_dict(),
                ckpt_dir / f"{args.model}_best.pt",
            )
            print(f"model is better than before, save model to {args.model}_best.pt")

        if loss > min_loss:
            early_stop += 1
        else:
            early_stop = 0
            min_loss = loss

        if early_stop == 10:
            print("Early stop...")
            break

    print(f"Done! Best model Acc: {(100 * max_acc):>.4f}%")
    torch.save(model.state_dict(), ckpt_dir / f"{args.model}.pt")

    with open("result_QA.txt", "a") as f:
        f.write(f"{backbone}/{args.model}, {max_acc:>5f}\n")
    with open(ckpt_dir / Path("learning_curve.json"), "w") as f:
        json.dump(learning_curve, f)


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
        default="./cache/QA",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/QA",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="model name.",
        default="model",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)
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

    # model
    parser.add_argument(
        "--backbone",
        help="bert backbone",
        default=None,
    )
    parser.add_argument(
        "--no_pretrained", help="do not use pretrained weight", action="store_true"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

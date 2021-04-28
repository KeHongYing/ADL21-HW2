import json
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


def iter_loop(dataloader, model, loss_fn, optimizer, device, mode):
    total_start_correct, total_end_correct = 0, 0
    total_sentence_correct = 0
    total_start_loss, total_end_loss = 0, 0
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

                start_correct, end_correct = 0, 0
                sentence_correct = 0
                cnt += 1

                token = data["token"].to(device)
                start = data["start"].to(device)
                end = data["end"].to(device)
                index = data["index"].to(device)

                pred = model(token)

                start_loss = loss_fn(pred["start"].transpose(1, 2), start)
                end_loss = loss_fn(pred["end"].transpose(1, 2), end)

                count = 0
                softmax = torch.nn.Softmax(dim=-1)
                for idx, (head, tail) in enumerate(index):
                    pred_start, pred_end = (
                        torch.zeros_like(pred["start"][idx][head:tail][:, 1]).to(
                            device
                        ),
                        torch.zeros_like(pred["end"][idx][head:tail][:, 1]).to(device),
                    )
                    pred_start[
                        softmax(pred["start"][idx][head:tail])[:, 1].argmax()
                    ] = 1
                    pred_end[softmax(pred["end"][idx][head:tail])[:, 1].argmax()] = 1

                    start_correct += (
                        (pred_start == start[idx][head:tail])
                        .type(torch.float)
                        .sum()
                        .item()
                    )
                    end_correct += (
                        (pred_end == end[idx][head:tail]).type(torch.float).sum().item()
                    )
                    sentence_correct += (
                        (
                            torch.all(
                                pred_start == start[idx][head:tail],
                                dim=-1,
                            )
                            and torch.all(
                                pred_end == end[idx][head:tail],
                                dim=-1,
                            )
                        )
                        .type(torch.int)
                        .item()
                    )
                    count += tail - head

                start_correct /= count
                end_correct /= count

                total_start_correct += start_correct
                total_end_correct += end_correct
                total_sentence_correct += sentence_correct
                total_start_loss += start_loss
                total_end_loss += end_loss

                sentence_correct /= pred["start"].shape[0]

                if mode == TRAIN:
                    optimizer.zero_grad()
                    (start_loss + end_loss).backward()
                    optimizer.step()

                current += token.shape[0]

                tepoch.set_postfix(
                    s_loss=f"{start_loss.item():>.4f}",
                    e_loss=f"{end_loss.item():>.4f}",
                    s_Acc=f"{start_correct:>.5f}",
                    e_Acc=f"{end_correct:>.5f}",
                    sent_Acc=f"{sentence_correct:>.2f}",
                )

            total_start_correct /= cnt
            total_end_correct /= cnt
            total_sentence_correct /= current
            total_start_loss /= cnt
            total_end_loss /= cnt
            print(
                "\033[2K\033[2K"
                + f"[{mode:>5}] "
                + f" start Acc: {total_start_correct:>.5f},"
                + f" end Acc: {total_end_correct:>.5f},"
                + f" sentence Acc: {total_sentence_correct:>.5f},"
                + f" start loss: {total_start_loss:>.4f},"
                + f" end loss: {total_end_loss:>.4f},",
            )

    return sentence_correct, total_start_loss + total_end_loss


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

    weights = torch.tensor([0.005, 0.995], dtype=torch.float).to(args.device)
    loss_fn = torch.torch.nn.CrossEntropyLoss(weight=weights)
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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

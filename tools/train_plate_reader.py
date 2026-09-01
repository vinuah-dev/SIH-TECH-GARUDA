"""Train a plate recogniser on the labelled crops, and say honestly whether it won.

The measurement in the README is clear about where reading stands: a general
scene-text recogniser gets 71% of characters on real Indian plates and slips on
same-class shapes - a `4` read as a `1` - which no amount of format repair can
separate. The only thing that moves those is a recogniser that has actually
seen this typeface at this size.

So this trains one. A CRNN read end to end with CTC: convolutions over the
crop, a recurrent pass along its width, and a loss that does not need to be
told where each character sits. No character segmentation, which on a blurred
plate is its own failure mode.

**It starts at a large disadvantage and that is worth stating up front.** There
are 1380 training crops here. The recogniser it is competing with was trained
on millions of images. A small model on a small dataset beating it would be a
surprise, and the honest outcome may well be that it does not - in which case
that is the result, and multi-frame consensus at 80% precision remains the
answer.

    python tools/train_plate_reader.py --epochs 120
    python tools/train_plate_reader.py --evaluate models/plate-reader.pt

The split is by plate number, not photograph (see tools/plate_dataset.py), so
a score here cannot come from having memorised a car.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0                                   # CTC reserves index 0
TO_INDEX = {c: i + 1 for i, c in enumerate(CHARS)}
TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}

HEIGHT, WIDTH = 32, 128
ROOT = Path("data/datasets/crops")
DEFAULT_MODEL = Path("models/plate-reader.pt")


# --------------------------------------------------------------------- data


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make a clean crop look like one a camera would actually deliver.

    With 1380 samples the model will memorise them in a few epochs unless every
    epoch shows it something new. These are not decorative: blur, low
    resolution and glare are precisely what the real failures look like, so
    training without them would produce a model that reads the dataset and not
    a road.
    """
    h, w = image.shape[:2]

    # A third of the time, show it the plate as it is.
    #
    # The first attempt at this applied every degradation below with high
    # probability, and the model then failed to fit even its own training data:
    # 12% exact on train, 49% of characters. That is not a shortage of data, it
    # is a shortage of *legible* data - stacked hard enough, the augmentations
    # destroyed the signal they were meant to make robust. A model must be able
    # to read a clean plate before being asked to read a ruined one.
    if rng.random() < 0.33:
        return image

    # Perspective: a plate is almost never square to the camera.
    if rng.random() < 0.35:
        shift = 0.06
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [rng.uniform(-shift, shift) * w, rng.uniform(-shift, shift) * h],
            [w + rng.uniform(-shift, shift) * w, rng.uniform(-shift, shift) * h],
            [w + rng.uniform(-shift, shift) * w, h + rng.uniform(-shift, shift) * h],
            [rng.uniform(-shift, shift) * w, h + rng.uniform(-shift, shift) * h],
        ])
        image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)

    # Distance: throw pixels away, then put the size back. This is the single
    # most important one - it is what a plate 40 px wide actually looks like.
    if rng.random() < 0.40:
        scale = rng.uniform(0.45, 0.90)
        small = cv2.resize(image, (max(8, int(w * scale)), max(4, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if rng.random() < 0.30:
        k = 3
        image = cv2.GaussianBlur(image, (k, k), 0)

    # Motion, in the direction a vehicle actually moves.
    if rng.random() < 0.12:
        length = rng.choice([3, 5])
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        image = cv2.filter2D(image, -1, kernel)

    if rng.random() < 0.45:
        alpha = rng.uniform(0.75, 1.3)         # contrast
        beta = rng.uniform(-25, 25)            # brightness, incl. glare and shade
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if rng.random() < 0.20:
        noise = np.random.normal(0, rng.uniform(3, 8), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.20:
        angle = rng.uniform(-3, 3)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return image


def prepare(image: np.ndarray) -> np.ndarray:
    """Grey, fixed size, normalised. The same path at train and test time."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return (image.astype(np.float32) / 127.5) - 1.0


class Plates(Dataset):
    def __init__(self, split: str, root: Path = ROOT, train: bool = False):
        self.root, self.train = root, train
        self.rows = [
            row for row in
            (json.loads(line) for line in
             (root / split / "labels.jsonl").read_text(encoding="utf-8").splitlines())
            if all(c in TO_INDEX for c in row["plate"]) and 4 <= len(row["plate"]) <= 12
        ]
        self.rng = random.Random(1234)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        image = cv2.imread(str(self.root / row["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((HEIGHT, WIDTH), np.uint8)
        if self.train:
            image = augment(image, self.rng)
        tensor = torch.from_numpy(prepare(image)).unsqueeze(0)
        target = torch.tensor([TO_INDEX[c] for c in row["plate"]], dtype=torch.long)
        return tensor, target, row["plate"]


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return images, targets, lengths, [b[2] for b in batch]


# -------------------------------------------------------------------- model


class PlateReader(nn.Module):
    """CNN down to a strip, then a recurrent pass along it, then CTC.

    Height is collapsed to 1 and width kept at 32 steps, so the recurrent layer
    reads left to right across the plate the way the number is written. Small
    on purpose: a larger model on 1380 crops memorises rather than learns.
    """

    def __init__(self, classes: int = len(CHARS) + 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                               # 16 x 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                               # 8 x 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                     # 4 x 32
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                     # 2 x 32
            nn.Conv2d(256, 256, (2, 1)), nn.BatchNorm2d(256), nn.ReLU(True),  # 1 x 32
        )
        self.rnn = nn.LSTM(256, 192, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.2)
        self.head = nn.Linear(384, classes)

    def forward(self, x):
        x = self.cnn(x)                     # B, 256, 1, 32
        x = x.squeeze(2).permute(0, 2, 1)   # B, 32, 256
        x, _ = self.rnn(x)
        return self.head(x)                 # B, 32, classes


def decode(logits) -> list[str]:
    """Greedy CTC: take the best class per step, drop repeats and blanks."""
    best = logits.argmax(dim=2).cpu().numpy()
    out = []
    for row in best:
        chars, previous = [], -1
        for index in row:
            if index != previous and index != BLANK:
                chars.append(TO_CHAR.get(int(index), ""))
            previous = index
        out.append("".join(chars))
    return out


# ----------------------------------------------------------------- scoring


def score(model, loader, device) -> dict:
    model.eval()
    exact = total = char_hit = char_total = 0
    wrong: list[tuple[str, str]] = []
    with torch.no_grad():
        for images, _, _, truths in loader:
            guesses = decode(model(images.to(device)))
            for truth, guess in zip(truths, guesses):
                total += 1
                if truth == guess:
                    exact += 1
                elif len(wrong) < 40:
                    wrong.append((truth, guess))
                char_total += len(truth)
                char_hit += sum(1 for a, b in zip(truth, guess) if a == b)
    return {
        "total": total,
        "exact": exact,
        "exact_pct": 100.0 * exact / max(total, 1),
        "char_pct": 100.0 * char_hit / max(char_total, 1),
        "wrong": wrong,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default=str(DEFAULT_MODEL))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--evaluate", default=None, help="score a saved model and stop")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    device = torch.device("cpu")
    root = Path(args.root)

    if not (root / "train" / "labels.jsonl").exists():
        print(f"no crops under {root} - run tools/plate_dataset.py first", file=sys.stderr)
        return 1

    train_set = Plates("train", root, train=True)
    val_set = Plates("val", root, train=False)
    val_loader = DataLoader(val_set, batch_size=64, collate_fn=collate)

    model = PlateReader().to(device)
    params = sum(p.numel() for p in model.parameters())

    if args.evaluate:
        model.load_state_dict(torch.load(args.evaluate, map_location=device))
        result = score(model, val_loader, device)
        print(f"  val exact : {result['exact']}/{result['total']} "
              f"({result['exact_pct']:.1f}%)")
        print(f"  val chars : {result['char_pct']:.1f}%")
        for truth, guess in result["wrong"][:12]:
            print(f"    {truth:<12} -> {guess}")
        return 0

    print(f"train    : {len(train_set)} crops")
    print(f"val      : {len(val_set)} crops (no plate number appears in both)")
    print(f"model    : CRNN + CTC, {params/1e6:.2f}M parameters, CPU")
    print(f"baseline : EasyOCR reads 34.5% of plates exactly, 71.2% of characters")
    print()

    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                        collate_fn=collate, num_workers=args.workers, drop_last=True)
    loss_fn = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr * 3, total_steps=args.epochs * len(loader),
        pct_start=0.2,
    )

    # Selected on exact reads, with character accuracy as the tie-break.
    #
    # Exact alone is not enough: CTC sits at 0% exact for many epochs while
    # character accuracy climbs steadily, and a checkpoint chosen on exact
    # alone keeps the first garbage model through all of it - then loses
    # everything learned since if the run is stopped.
    best = {"exact_pct": -1.0, "char_pct": -1.0}
    started = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets, lengths, _ in loader:
            images = images.to(device)
            logits = model(images)
            logp = logits.log_softmax(2).permute(1, 0, 2)     # T, B, C
            input_lengths = torch.full((images.size(0),), logits.size(1),
                                       dtype=torch.long)
            loss = loss_fn(logp, targets, input_lengths, lengths)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            schedule.step()
            running += loss.item()

        if epoch % 5 == 0 or epoch == args.epochs:
            result = score(model, val_loader, device)
            mark = ""
            better = (result["exact_pct"], result["char_pct"]) > (
                best["exact_pct"], best["char_pct"]
            )
            if better:
                best = result | {"epoch": epoch}
                torch.save(model.state_dict(), args.out)
                mark = "  <- saved"
            print(f"  epoch {epoch:>3}  loss {running/max(len(loader),1):6.3f}  "
                  f"val exact {result['exact_pct']:5.1f}%  "
                  f"chars {result['char_pct']:5.1f}%{mark}")

    elapsed = time.time() - started
    print()
    print("=" * 62)
    print(f"  trained            : {args.epochs} epochs in {elapsed/60:.1f} min")
    print(f"  best val exact     : {best['exact_pct']:.1f}%  (epoch {best.get('epoch')})")
    print(f"  best val characters: {best['char_pct']:.1f}%")
    print(f"  EasyOCR, same task : 34.5% exact, 71.2% characters")
    print()
    if best["exact_pct"] > 34.5:
        print("  This model reads more plates than the general recogniser does.")
        print("  Wire it in behind the same consensus vote and re-run the benchmark;")
        print("  a win here still has to survive that before it means anything.")
    else:
        print("  It did NOT beat the general recogniser. That is a real result, not")
        print("  a setup to be tuned away: 1380 crops is very little against the")
        print("  millions EasyOCR saw. Multi-frame consensus at 80% precision")
        print("  remains the better answer until there is more data.")
    print("=" * 62)
    if best["wrong"]:
        print()
        print(f"  {'truth':<14}{'read as':<16}")
        for truth, guess in best["wrong"][:12]:
            print(f"  {truth:<14}{guess or '(nothing)':<16}")
    print(f"\n  saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

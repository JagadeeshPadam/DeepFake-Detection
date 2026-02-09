# Generated from: final.ipynb
# Converted at: 2026-02-09T17:00:11.035Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install timm --no-deps -q

import os
from pathlib import Path
import random
import numpy as np

import cv2
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

DATA_ROOT = Path("/kaggle/input/deep-fake-detection-dfd-entire-original-dataset")
OUT_ROOT  = Path("/kaggle/working/dfd_faces")  

NUM_FRAMES = 16
IMG_SIZE   = 224
BATCH_SIZE = 4
EPOCHS     = 25
LR         = 1e-4
SEED       = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
OUT_ROOT.mkdir(parents=True, exist_ok=True)


real_dir = DATA_ROOT / "DFD_original sequences"
fake_dir = DATA_ROOT / "DFD_manipulated_sequences" / "DFD_manipulated_sequences"

real_videos = sorted([p for p in real_dir.glob("*.mp4")])
fake_videos = sorted([p for p in fake_dir.glob("*.mp4")])[:400]

print("Num real:", len(real_videos))
print("Num fake:", len(fake_videos))

all_paths = real_videos + fake_videos
all_labels = [1]*len(real_videos) + [0]*len(fake_videos)   # 1: real, 0: fake

train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=SEED, stratify=train_labels
)

def build_split_list(paths, labels):
    return [{"path": p, "label": int(l)} for p, l in zip(paths, labels)]

splits = {
    "train": build_split_list(train_paths, train_labels),
    "val":   build_split_list(val_paths,   val_labels),
    "test":  build_split_list(test_paths,  test_labels),
}

for k,v in splits.items():
    print(k, len(v))


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def center_crop_resize(img, size=224):
    h, w = img.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    crop = img[y1:y1+side, x1:x1+side]
    return cv2.resize(crop, (size, size))

def detect_face_crop(img, size=224):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                          minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return center_crop_resize(img, size), False

    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    x1, y1, x2, y2 = x, y, x+w, y+h
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (size, size))
    return face, True

def extract_faces_from_video(video_path: Path, out_dir: Path,
                             num_frames=16, img_size=224):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Cannot open:", video_path)
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("No frames:", video_path)
        cap.release()
        return 0

    indices = np.linspace(0, total-1, num_frames, dtype=int)

    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        face, _ = detect_face_crop(frame, size=img_size)
        out_path = out_dir / f"{i:03d}.jpg"
        cv2.imwrite(str(out_path), face)
        saved += 1

    cap.release()
    return saved


def get_class_name(label_int):
    return "real" if label_int == 1 else "fake"

for split_name, items in splits.items():
    print("Processing", split_name, "...")
    for info in items:
        vpath = info["path"]
        label = info["label"]
        cls_name = get_class_name(label)

        video_id = vpath.stem  
        out_dir = OUT_ROOT / split_name / cls_name / video_id

        if out_dir.exists():
            continue

        n_saved = extract_faces_from_video(vpath, out_dir,
                                           num_frames=NUM_FRAMES,
                                           img_size=IMG_SIZE)
        if n_saved == 0:
            if out_dir.exists():
                for f in out_dir.glob("*.jpg"):
                    f.unlink()
                out_dir.rmdir()
    print("Done", split_name)

print("Preprocessing finished. Root:", OUT_ROOT)


from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch

class VideoFaceDataset(Dataset):
    def __init__(self, root, num_frames=16, img_size=224):
        self.root = Path(root)
        self.num_frames = num_frames

        self.classes = ["fake", "real"]
        self.class_to_idx = {"fake": 0, "real": 1}

        self.samples = []
        for cls in self.classes:
            cdir = self.root / cls
            if not cdir.exists():
                continue
            for video_dir in sorted(cdir.iterdir()):
                if video_dir.is_dir():
                    frames = sorted(video_dir.glob("*.jpg"))
                    if len(frames) == 0:
                        continue
                    self.samples.append((video_dir, self.class_to_idx[cls]))

        print(f"Dataset @ {root}: {len(self.samples)} clips")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, frame_paths):
        n = len(frame_paths)
        if n >= self.num_frames:
            idx = np.linspace(0, n-1, self.num_frames, dtype=int)
        else:
            idx = list(range(n)) + [n-1]*(self.num_frames-n)
        return [frame_paths[i] for i in idx]

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frames = sorted(video_dir.glob("*.jpg"))
        frames = self._sample_frames(frames)

        imgs = []
        for p in frames:
            img = Image.open(p).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        frames_tensor = torch.stack(imgs, dim=0)  # (T,C,H,W)
        return frames_tensor, torch.tensor(label, dtype=torch.long)

FACE_ROOT = OUT_ROOT   

NUM_FRAMES = 16
IMG_SIZE   = 224
BATCH_SIZE = 8

train_img_ds = VideoFaceDataset(FACE_ROOT / "train", NUM_FRAMES, IMG_SIZE)
val_img_ds   = VideoFaceDataset(FACE_ROOT / "val",   NUM_FRAMES, IMG_SIZE)
test_img_ds  = VideoFaceDataset(FACE_ROOT / "test",  NUM_FRAMES, IMG_SIZE)

train_img_loader = DataLoader(train_img_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
val_img_loader   = DataLoader(val_img_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
test_img_loader  = DataLoader(test_img_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)


import timm
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

def get_effnet_extractor(model_name="tf_efficientnet_b0_ns"):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,   
        global_pool="avg"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

effnet = get_effnet_extractor().to(device)
FEATURE_DIM = effnet.num_features
print("Feature dim:", FEATURE_DIM)


from tqdm import tqdm

@torch.no_grad()
def extract_features_split(loader, save_path):
    all_feats = []
    all_labels = []

    for frames, labels in tqdm(loader, desc=f"Extracting -> {save_path}"):
        #
        B, T, C, H, W = frames.shape
        frames = frames.to(device)

        frames_flat = frames.view(B*T, C, H, W)    # (B*T,C,H,W)
        feats_flat = effnet(frames_flat)           # (B*T,F)
        feats = feats_flat.view(B, T, -1)          # (B,T,F)

        all_feats.append(feats.cpu())
        all_labels.append(labels.clone())

    X = torch.cat(all_feats, dim=0)   # (N,T,F)
    y = torch.cat(all_labels, dim=0)  # (N,)
    torch.save({"features": X, "labels": y}, save_path)
    print(f"Saved {save_path}: X={X.shape}, y={y.shape}")

SAVE_ROOT = Path("/kaggle/working/dfd_effnet_features")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

extract_features_split(train_img_loader, SAVE_ROOT / "train_features.pt")
extract_features_split(val_img_loader,   SAVE_ROOT / "val_features.pt")
extract_features_split(test_img_loader,  SAVE_ROOT / "test_features.pt")


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.X = data["features"]  # (N,T,F)
        self.y = data["labels"]    # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_feat_ds = FeatureDataset(SAVE_ROOT / "train_features.pt")
val_feat_ds   = FeatureDataset(SAVE_ROOT / "val_features.pt")
test_feat_ds  = FeatureDataset(SAVE_ROOT / "test_features.pt")

train_feat_loader = DataLoader(train_feat_ds, batch_size=64, shuffle=True)
val_feat_loader   = DataLoader(val_feat_ds,   batch_size=64, shuffle=False)
test_feat_loader  = DataLoader(test_feat_ds,  batch_size=64, shuffle=False)

print("Train features:", train_feat_ds.X.shape)


import math
import torch.nn.functional as F

class TemporalViT(nn.Module):
    def __init__(self,
                 feature_dim=FEATURE_DIM,
                 d_model=256,
                 n_heads=4,
                 num_layers=4,
                 num_classes=2,
                 dropout=0.1,
                 max_len=64):
        super().__init__()

        self.proj = nn.Linear(feature_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len+1, d_model))  # +1 cho CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B,T,F)
        B, T, Fd = x.shape
        x = self.proj(x)                   # (B,T,d_model)

        cls = self.cls_token.expand(B, -1, -1)  # (B,1,d_model)
        x = torch.cat([cls, x], dim=1)          # (B,T+1,d_model)

        x = x + self.pos_embed[:, :T+1, :]      # add positional encoding

        h = self.encoder(x)                     # (B,T+1,d_model)
        cls_out = h[:, 0]                     
        cls_out = self.norm(cls_out)
        logits = self.fc(cls_out)
        return logits

num_classes = 2
vit_model = TemporalViT(
    feature_dim=FEATURE_DIM,
    d_model=256,
    n_heads=4,
    num_layers=4,
    num_classes=num_classes,
    max_len=NUM_FRAMES
).to(device)

print(vit_model)


def get_class_weights_from_feat_ds(dataset):
    labels = dataset.y
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (counts * len(counts) + 1e-8)
    return weights

class_weights = get_class_weights_from_feat_ds(train_feat_ds).to(device)
print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(vit_model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

def train_epoch():
    vit_model.train()
    total_loss, correct, total = 0., 0, 0
    for X, y in train_feat_loader:
        X, y = X.to(device), y.to(device)   # X: (B,T,F)
        logits = vit_model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss/len(train_feat_loader), correct/total

@torch.no_grad()
def eval_epoch(loader):
    vit_model.eval()
    total_loss, correct, total = 0., 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = vit_model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss/len(loader), correct/total

EPOCHS = 60
best_val = 0.0
best_path = "/kaggle/working/best_temporal_vit.pth"

hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc = train_epoch()
    val_loss, val_acc = eval_epoch(val_feat_loader)
    scheduler.step()

    hist["train_loss"].append(tr_loss)
    hist["val_loss"].append(val_loss)
    hist["train_acc"].append(tr_acc)
    hist["val_acc"].append(val_acc)

    if val_acc > best_val:
        best_val = val_acc
        torch.save(vit_model.state_dict(), best_path)
        print(f"[Epoch {ep:02d}] New best val_acc={val_acc*100:.2f}%, saved!")

    print(f"Epoch {ep:02d} | "
          f"Train {tr_loss:.4f}/{tr_acc*100:.2f}% | "
          f"Val {val_loss:.4f}/{val_acc*100:.2f}%")

print("Best Val Acc:", best_val*100)


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

vit_model.load_state_dict(torch.load(best_path))
vit_model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for X, y in test_feat_loader:
        X = X.to(device)
        logits = vit_model(X)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())

# Precision, Recall, F1-score
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds,
                            target_names=["Fake", "Real"]))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# ROC–AUC
auc = roc_auc_score(all_labels, all_probs)
print("ROC-AUC:", auc)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
            cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]

plt.figure(figsize=(5,4))
sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
            cmap="Greens")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(all_labels, all_probs)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.tight_layout()
plt.show()


epochs = range(1, len(hist["train_loss"]) + 1)

plt.figure(figsize=(6,4))
plt.plot(epochs, hist["train_loss"], label="Training Loss")
plt.plot(epochs, hist["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.plot(epochs, hist["train_acc"], label="Training Accuracy")
plt.plot(epochs, hist["val_acc"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.hist(all_probs, bins=30)

plt.xlabel("Predicted Probability (Real)")
plt.ylabel("Frequency")
plt.title("Prediction Probability Distribution")
plt.tight_layout()
plt.show()


# ==============================================================================
# DOWNLOAD AND DEPLOYMENT SETUP (SHOW FILES + AUTO DOWNLOAD)
# ==============================================================================

import json
from IPython.display import display, Javascript, Markdown
from IPython.display import FileLink

# 1. Export Model Metadata
metadata = {
    "IMG_SIZE": IMG_SIZE,
    "NUM_FRAMES": NUM_FRAMES,
    "FEATURE_DIM": FEATURE_DIM,
    "ARCH": "tf_efficientnet_b0_ns + TemporalViT",
    "CLASSES": ["Fake", "Real"]
}

with open("/kaggle/working/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# 2. Files to download
files = [
    "best_temporal_vit.pth",
    "model_metadata.json"
]

# 3. SHOW files clearly in output
display(Markdown("### 📦 Files prepared for download:"))
for f in files:
    display(Markdown(f"- `{f}`"))

# 4. AUTO-DOWNLOAD (no click required)
js_code = """
(function() {
    const files = %s;
    files.forEach((file, i) => {
        setTimeout(() => {
            const link = document.createElement('a');
            link.href = '/kaggle/working/' + file;
            link.download = file;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }, i * 800);  // delay to avoid browser blocking
    });
})();
""" % files

display(Javascript(js_code))

display(FileLink(r'best_temporal_vit.pth'))
display(FileLink(r'model_metadata.json'))
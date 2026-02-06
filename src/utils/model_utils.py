import torch
from PIL import Image
from torch.utils.data import Dataset

# -------------------------
# Dataset
# -------------------------

class FaceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["face_path"].to_list()
        self.labels = df["label"].to_list()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
    
@torch.no_grad()
def evaluate_with_outputs(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.append(y.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    return {
        "labels": torch.cat(all_labels),
        "preds": torch.cat(all_preds),
        "probs": torch.cat(all_probs),
    }


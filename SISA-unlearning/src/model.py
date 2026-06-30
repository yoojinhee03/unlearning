"""모델 정의. SISA의 각 shard는 독립된 모델 인스턴스를 사용한다."""

from pathlib import Path

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """SVHN 분류용 3층 CNN (Conv2d 3층 + FC 2층)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_model(arch: str = "simple_cnn", num_classes: int = 10) -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {arch!r}")


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    shard_index: int,
    slice_index: int,
    epoch: int,
) -> None:
    """모델 가중치와 메타데이터를 함께 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "shard_index": shard_index,
            "slice_index": slice_index,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
) -> dict:
    """체크포인트를 로드해 model에 가중치를 적용하고 메타데이터를 반환."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


# ── 간편 저장/로드 (메타데이터 없이 state dict만) ────────────────────────────

def save_model(model: nn.Module, path: str | Path) -> None:
    """모델 state dict만 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path: str | Path,
    arch: str = "simple_cnn",
    num_classes: int = 10,
    device: torch.device | None = None,
) -> nn.Module:
    """저장된 가중치에서 모델을 복원해 반환.

    save_model()로 저장한 bare state dict와
    save_checkpoint()로 저장한 dict 형식 모두 처리한다.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(arch=arch, num_classes=num_classes).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model
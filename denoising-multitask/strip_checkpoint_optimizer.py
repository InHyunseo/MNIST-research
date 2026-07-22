"""기존 checkpoint에서 사용하지 않는 optimizer_state_dict를 제거하는 1회성 스크립트다.

학습은 30 epoch을 한 번에 끝내고 재개하지 않으므로 optimizer state는 재사용·평가
어디에서도 읽히지 않는다. 각 .pt 크기의 약 2/3를 차지하므로 제거해 disk를 회수한다.

torch가 필요하다. outputs/checkpoints/*.pt가 있고 torch가 설치된 환경(서버 등)에서
한 번 실행한 뒤 이 스크립트는 삭제한다. 재학습하지 않는다.

    python strip_checkpoint_optimizer.py

각 checkpoint는 임시 파일에 저장 후 원자 교체하며, 교체 직후 다시 로딩해
model_state_dict·configuration·training_complete가 온전한지 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import torch


CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "outputs" / "checkpoints"
REQUIRED_KEYS = ("model_state_dict", "configuration", "training_complete")


def strip_optimizer_state(checkpoint_path: Path) -> int:
    """한 checkpoint에서 optimizer_state_dict를 제거하고 회수한 byte 수를 반환한다."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "optimizer_state_dict" not in checkpoint:
        return 0
    original_size = checkpoint_path.stat().st_size
    del checkpoint["optimizer_state_dict"]

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)

    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing = [key for key in REQUIRED_KEYS if key not in reloaded]
    if missing:
        raise RuntimeError(f"검증 실패: {checkpoint_path}에 {missing} 키가 없습니다.")
    if "optimizer_state_dict" in reloaded:
        raise RuntimeError(f"검증 실패: {checkpoint_path}에 optimizer state가 남아 있습니다.")
    return original_size - checkpoint_path.stat().st_size


def main() -> None:
    if not CHECKPOINT_DIRECTORY.exists():
        raise FileNotFoundError(f"checkpoint 디렉터리가 없습니다: {CHECKPOINT_DIRECTORY}")
    checkpoint_paths = sorted(CHECKPOINT_DIRECTORY.glob("*.pt"))
    stripped_count = 0
    reclaimed_bytes = 0
    for checkpoint_path in checkpoint_paths:
        saved_bytes = strip_optimizer_state(checkpoint_path)
        if saved_bytes > 0:
            stripped_count += 1
            reclaimed_bytes += saved_bytes
    print(
        f"완료: {stripped_count}/{len(checkpoint_paths)}개 checkpoint에서 "
        f"optimizer state 제거, {reclaimed_bytes / (1024 * 1024):.1f}MB 회수"
    )


if __name__ == "__main__":
    main()

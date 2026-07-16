"""MNIST test set을 data/*.u8 raw uint8로 덤프 (C++가 동일 바이트 사용)."""
from mnist_core.dataset import dump_u8

if __name__ == "__main__":
    print("dumped", dump_u8())

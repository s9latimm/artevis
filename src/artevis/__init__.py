import torch

DPI: int = 100
SCALE: float = 2.

SIZE: int = 256
N: int = 50_000
THRESHOLD: int = 1
CUDA: bool = True
CACHE: bool = True
DTYPE: torch.dtype = torch.float32

PROJECTS: [str] = ['mona', 'pearl', 'fog', 'scream', 'stars'][:1]
FPS: float = .6

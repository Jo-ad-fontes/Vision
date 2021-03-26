from dataclasses import dataclass
# init 자동생성


@dataclass
class Config:
    device: str
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float

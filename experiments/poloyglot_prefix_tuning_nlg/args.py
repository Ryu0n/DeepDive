from dataclasses import dataclass


@dataclass(frozen=True)
class Args:
    # pretrained_model_checkpoint: str = "EleutherAI/polyglot-ko-1.3b"
    pretrained_model_checkpoint: str = "EleutherAI/polyglot-ko-3.8b"
    special_tokens = None
    freeze_plm: bool = True
    freeze_prefix: bool = False
    prefix_dropout: float = 0.1
    prefix_sequence_length: int = 8
    mid_dim: int = 800
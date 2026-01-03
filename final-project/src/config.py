"""Central configuration for the ROBUST04 ranking competition."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_all_seeds(seed: int = 42) -> None:
    """Ensure reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class PathConfig:
    """File and directory paths."""
    project_root: Path = PROJECT_ROOT
    
    # Input data
    queries_file: Path = PROJECT_ROOT / "Files-20260103" / "queriesROBUST.txt"
    qrels_file: Path = PROJECT_ROOT / "Files-20260103" / "qrels_50_Queries"
    expanded_queries_file: Path = PROJECT_ROOT / "data" / "expanded_queries.csv"
    
    # Output directories
    results_dir: Path = PROJECT_ROOT / "results"
    tuning_dir: Path = PROJECT_ROOT / "tuning_results"
    
    # Output files
    run1_file: Path = results_dir / "run_1.res"
    run2_file: Path = results_dir / "run_2.res"
    run3_file: Path = results_dir / "run_3.res"


@dataclass
class BM25Config:
    """BM25 retrieval parameters."""
    # Index name (pyserini prebuilt)
    index_name: str = "robust04"
    
    # BM25 parameters (to be tuned)
    k1: float = 0.9
    b: float = 0.4
    
    # Search depth
    num_results: int = 1000
    
    # Grid search ranges
    k1_range: list[float] = field(default_factory=lambda: [0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6])
    b_range: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8])


@dataclass
class RM3Config:
    """RM3 pseudo-relevance feedback parameters."""
    # RM3 parameters (to be tuned)
    fb_docs: int = 10
    fb_terms: int = 10
    original_query_weight: float = 0.5
    
    # Grid search ranges
    fb_docs_range: list[int] = field(default_factory=lambda: [3, 5, 10, 15, 20])
    fb_terms_range: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    original_weight_range: list[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


@dataclass
class ChunkingConfig:
    """Document chunking parameters."""
    chunk_size: int = 256  # characters
    chunk_overlap: int = 64
    
    # Separators for RecursiveCharacterTextSplitter (priority order)
    separators: list[str] = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""
    ])


@dataclass
class NeuralConfig:
    """Neural reranking parameters - MAXED OUT FOR A100 80GB."""
    # Bi-Encoder - fast first stage filtering
    bi_encoder_model: str = "BAAI/bge-large-en-v1.5"  # Best English (~1.3GB)
    bi_encoder_batch_size: int = 1024  # Bi-encoder is VERY fast, max it!
    bi_encoder_top_k: int = 500  # Filter to top 500 before cross-encoder
    
    # Cross-Encoder - USE BGE-RERANKER (state-of-the-art!)
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"  # Best quality (~2.3GB)
    cross_encoder_batch_size: int = 1024  # A100 80GB can handle this!
    cross_encoder_max_length: int = 512
    
    # MonoT5 - USE 3B MODEL for best quality
    monot5_model: str = "castorini/monot5-3b-msmarco"  # 3B params, ~12GB VRAM
    monot5_batch_size: int = 256  # A100 80GB can do 256 batches!
    monot5_max_length: int = 512
    
    # Ensemble weight - BGE and MonoT5-3B are both strong, balanced weight
    ce_weight: float = 0.5  # Equal weight since both are state-of-the-art
    
    # Reranking depth - rerank MORE docs for better recall
    rerank_depth: int = 200  # Top-k docs from BM25 to rerank
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Precision - BF16 for A100 (native support)
    use_bf16: bool = True
    
    # Parallel processing
    num_workers: int = 8  # Original working value


@dataclass
class AggregationConfig:
    """Score aggregation parameters."""
    # Strategy: 'maxp', 'sump', 'firstp', 'topkp'
    strategy: str = "maxp"
    
    # For topkp strategy
    top_k: int = 3


@dataclass
class FusionConfig:
    """Multi-signal fusion parameters."""
    # RRF parameter
    rrf_k: int = 60
    
    # Grid search range for k
    rrf_k_range: list[int] = field(default_factory=lambda: [40, 50, 60, 70, 80])
    
    # Fusion method: 'rrf', 'combsum', 'combmnz', 'weighted'
    method: str = "rrf"
    
    # Weights for weighted fusion (will be tuned)
    # Order: bm25, bm25_rm3, bm25_q2d, neural
    weights: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.2, 0.5])


@dataclass
class TuningConfig:
    """Parameter tuning configuration."""
    # Cross-validation
    n_folds: int = 5
    random_seed: int = 42
    
    # Train query IDs (301-350)
    train_qids: list[str] = field(default_factory=lambda: [str(i) for i in range(301, 351)])


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    rm3: RM3Config = field(default_factory=RM3Config)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    
    def __post_init__(self):
        # Ensure output directories exist
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.paths.tuning_dir.mkdir(parents=True, exist_ok=True)


# Global default config
DEFAULT_CONFIG = Config()


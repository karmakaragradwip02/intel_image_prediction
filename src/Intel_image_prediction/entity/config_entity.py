from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen= True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen= True)
class DataPreparationConfig:
    root_dir: Path
    data_dir: Path
    train_dir: Path
    test_dir: Path
    val_dir: Path

@dataclass(frozen= True)
class ModelPreparationTrainingConfig:
    root_dir: Path
    model_dir: Path
    train_dir: Path
    val_dir: Path
    history_dir: Path
    learning_rate: float
    classes: int
    epochs: int
    weight_decay: float
    input_image_size: list
    epsilon: float
    momentum: float
    decay_rate: float
    batch_size: int

@dataclass(frozen= True)
class ModelEvaluationConfig:
    root_dir: Path
    trained_model_dir: Path
    test_dir: Path
    history_dir: Path
    graph_dir: Path
    mlflow_uri: str
    all_params: dict
    epochs: int
    classes: int
from Intel_image_prediction.constants import *
from Intel_image_prediction.utils.common import read_yaml, create_directories
from Intel_image_prediction.entity.config_entity import DataIngestionConfig, DataPreparationConfig, ModelPreparationTrainingConfig, ModelEvaluationConfig


class ConfigurationManager:
    def __init__(self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_cofig = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_cofig
    
    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation

        create_directories([config.root_dir])

        data_preparation_config = DataPreparationConfig(
            root_dir = config.root_dir,
            data_dir = config.data_dir,
            train_dir = config.train_dir,
            test_dir = config.test_dir,
            val_dir = config.val_dir
        )

        return data_preparation_config
    
    def get_model_prep_train_config(self) -> ModelPreparationTrainingConfig:
        config = self.config.model_preparation_training
        create_directories([config.root_dir])

        model_prep_train_config = ModelPreparationTrainingConfig(
            root_dir=Path(config.root_dir),
            model_dir=Path(config.model_dir),
            train_dir=Path(config.train_dir),
            val_dir=Path(config.val_dir),
            history_dir=Path(config.history_dir),
            classes=int(self.params.classes),
            learning_rate=float(self.params.learning_rate),
            epochs=int(self.params.epochs),
            weight_decay=float(self.params.weight_decay),
            input_image_size=[int(x) for x in self.params.input_image_size],
            epsilon=float(self.params.epsilon),
            momentum = float(self.params.momentum),
            decay_rate = float(self.params.decay_rate),
            batch_size=int(self.params.batch_size)
        )
        
        return model_prep_train_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = Path(config.root_dir),
            trained_model_dir = Path(config.trained_model_dir),
            history_dir= Path(config.history_dir),
            graph_dir = Path(config.graph_dir),
            test_dir= Path(config.test_dir),
            mlflow_uri="https://dagshub.com/karmakaragradwip02/intel_image_prediction.mlflow",
            all_params=self.params,
            epochs = self.params.epochs,
            classes = self.params.classes
        )

        return model_evaluation_config
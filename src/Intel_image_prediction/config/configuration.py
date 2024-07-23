from Intel_image_prediction.constants import *
from Intel_image_prediction.utils.common import read_yaml, create_directories
from Intel_image_prediction.entity.config_entity import DataIngestionConfig, DataPreparationConfig

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
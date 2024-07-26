from Intel_image_prediction import logger
from Intel_image_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from Intel_image_prediction.pipeline.data_preparation_pipeline import DataPreparationTrainingPipeline
from Intel_image_prediction.pipeline.model_prep_train_pipeline import ModelPreparationTrainingPipeline

STAGE_NAME = "DATA INGESTION STAGE"

try:
    logger.info(f"********************") 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"********************")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "DATA PREPARATION STAGE"

try:
    logger.info(f"********************") 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataPreparationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"********************")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "DATA TRAINING STAGE"

if __name__ == '__main__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
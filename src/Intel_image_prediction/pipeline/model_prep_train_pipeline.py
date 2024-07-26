from Intel_image_prediction.config.configuration import ConfigurationManager
from Intel_image_prediction.components.model_prep_train import ModelPreparation
from Intel_image_prediction import logger

STAGE_NAME = "DATA TRAINING STAGE"

class ModelPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_preparation_training_config = config.get_model_prep_train_config()
        model_preparation = ModelPreparation(config=model_preparation_training_config)
        train_loader, val_loader, train_count, val_count = model_preparation.image_processing()
        cnn = model_preparation.model()
        model, optimizer, scheduler, criterion = model_preparation.model_compilation(cnn)
        model_preparation.print_model_summary(cnn, (3, 150, 150))
        model = model_preparation.train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, train_count, val_count)
        model_preparation.save_model(model)
    
if __name__ == '__main__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
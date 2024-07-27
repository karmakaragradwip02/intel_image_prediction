from Intel_image_prediction.config.configuration import ConfigurationManager
from Intel_image_prediction.components.model_evaluation import ModelEvaluation
from Intel_image_prediction import logger

STAGE_NAME = "MODEL EVALUATION STAGE"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(model_evaluation_config)
        cnn = model_evaluation.model()
        transformer = model_evaluation.define_transforms()
        test_loader = model_evaluation.data_loader(transformer)
        model = model_evaluation.load_model(cnn)
        model_evaluation.log_into_mlflow(model, test_loader)
    

if __name__ == '__main__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
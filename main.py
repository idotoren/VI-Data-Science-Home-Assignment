
class Main:
    def __init__(self, data_folder, preprocessor_class, trainer_class, evaluator_class):
        """
        Initializes the main pipeline with the required components.

        Args:
            data_folder (str): Path to the folder containing the data files.
            preprocessor_class (class): The class responsible for data preprocessing.
            trainer_class (class): The class responsible for model training.
            evaluator_class (class): The class responsible for model evaluation.
        """
        self.data_folder = data_folder
        self.preprocessor_class = preprocessor_class
        self.trainer_class = trainer_class
        self.evaluator_class = evaluator_class

    def run(self):
        """
        Executes the full pipeline: data collection, preprocessing, model training, and evaluation.
        """
        print("Starting data collection...")
        data_collector = DataCollector(self.data_folder)
        raw_data = data_collector.run()

        print("Starting data preprocessing...")
        preprocessor = self.preprocessor_class()
        processed_data = preprocessor.run(raw_data)

        print("Starting model training...")
        trainer = self.trainer_class()
        predictions_df = trainer.train(processed_data)

        print("Starting results preparation...")
        evaluator = self.evaluator_class()
        evaluation_results = evaluator.run(predictions_df)

        print("Pipeline completed.")
        return evaluation_results


# Example usage
if __name__ == "__main__":
    from Data_collection.data_collection import DataCollector
    from Data_preparation.data_preprocessing import DataPreprocessing
    from Model_training.train import ModelTraining
    from Evaluate_model.evaluate_model import ModelEvaluation

    data_folder = "Data"
    main_pipeline = Main(
        data_folder=data_folder,
        preprocessor_class=DataPreprocessing,
        trainer_class=ModelTraining,
        evaluator_class=ModelEvaluation,
    )
    results = main_pipeline.run()
    print("Evaluation Results:", results)


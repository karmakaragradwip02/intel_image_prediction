import os
import sys
import time
import threading
import shutil
from sklearn.model_selection import train_test_split
from Intel_image_prediction import logger
from Intel_image_prediction.entity.config_entity import DataPreparationConfig

class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.stop_animation = False
        self.current_message = ""
        self.animation_lock = threading.Lock()
    
    def making_traintest_folder(self):
        try:
            train_folder = self.config.train_dir
            test_folder = self.config.test_dir
            val_folder = self.config.val_dir
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)
            logger.info("Created validation, test, and train folders")
        except Exception as e:
            raise e

    def animate(self):
        symbols = ['-', '\\', '|', '/']
        i = 0
        while not self.stop_animation:
            with self.animation_lock:
                message = self.current_message
            sys.stdout.write(f"\rSplitting the Data {symbols[i % len(symbols)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def split_data(self):
        data_path = self.config.data_dir
        class_names = os.listdir(data_path)
        train_folder = self.config.train_dir
        test_folder = self.config.test_dir
        val_folder = self.config.val_dir

        self.stop_animation = False
        animation_thread = threading.Thread(target=self.animate)
        animation_thread.start()

        try:
            for class_name in class_names:
                class_path = os.path.join(data_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                # List all files in the class directory
                files = os.listdir(class_path)
                files = [os.path.join(class_path, f) for f in files if os.path.isfile(os.path.join(class_path, f))]

                # Split the files into training and the remaining set
                train_files, remaining_files = train_test_split(files, test_size=0.3, random_state=42)
                # Split the remaining files into validation and testing sets
                val_files, test_files = train_test_split(remaining_files, test_size=1/3, random_state=42)

                # Create class directories in train, validation, and test folders
                train_class_folder = os.path.join(train_folder, class_name)
                val_class_folder = os.path.join(val_folder, class_name)
                test_class_folder = os.path.join(test_folder, class_name)
                os.makedirs(train_class_folder, exist_ok=True)
                os.makedirs(val_class_folder, exist_ok=True)
                os.makedirs(test_class_folder, exist_ok=True)

                # Move the files to the respective directories
                for file in train_files:
                    shutil.copy(file, train_class_folder)
                for file in val_files:
                    shutil.copy(file, val_class_folder)
                for file in test_files:
                    shutil.copy(file, test_class_folder)
        
            logger.info("The data has been split into train, validation, and test sets")
        finally:
            self.stop_animation = True
            animation_thread.join()
            sys.stdout.write("\rSplitting data complete.          \n")
            sys.stdout.flush()
from abc import ABC, abstractmethod

class BaseVideoEncoder(ABC):
    """
    Abstract base class for video encoders.
    Defines the basic interface and common functionalities for all encoders.
    """

    def __init__(self, **kwargs):
        """
        Initialize the video encoder.
        
        :param kwargs: Additional parameters for the encoder.
        """
        self.params = kwargs

    
    @abstractmethod
    def build_model(self):
        pass


    def save_model(self, file_path:str):
        """
        Save the trained model to a file.

        :param file_path: Path to save the model.
        """
        # Implementation for saving model
        pass

    def load_model(self, file_path:str):
        """
        Load a model from a file.

        :param file_path: Path to load the model from.
        """
        # Implementation for loading model
        pass

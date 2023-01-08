from bdilab_server.base.model import CEModel
from bdilab_server.base.storage import download_model
# from bdilab_detect.utils.saving import load_detector, Data
from bdilab_detect.utils.saving import load_detector


class BdilabDetectModel(CEModel):  # pylint:disable=c-extension-no-member
    # def __init__(self, name: str, storage_uri: str, model: Optional[Data] = None):
    def __init__(self, name: str, storage_uri: str, model = None):
        """
        Outlier Detection Model

        Parameters
        ----------
        name
             The name of the model
        storage_uri
             The URI location of the model
        """
        super().__init__(name)
        self.name = name
        self.storage_uri = storage_uri
        self.ready = False
        # self.model: Optional[Data] = model
        self.model = model

    def load(self):
        """
        Load the model from storage

        """
        model_folder = download_model(self.storage_uri)
        # self.model: Data = load_detector(model_folder)
        self.model = load_detector(model_folder)
        self.ready = True

from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class FasterRCNN(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from FasterRCNN.FasterRCNN_process import FasterRCNNProcessFactory
        # Instantiate process object
        return FasterRCNNProcessFactory()

    def getWidgetFactory(self):
        from FasterRCNN.FasterRCNN_widget import FasterRCNNWidgetFactory
        # Instantiate associated widget object
        return FasterRCNNWidgetFactory()

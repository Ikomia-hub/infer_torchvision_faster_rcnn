from ikomia import dataprocess
import FasterRCNN_process as processMod
import FasterRCNN_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class FasterRCNN(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.FasterRCNNProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.FasterRCNNWidgetFactory()

from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_torchvision_faster_rcnn.infer_torchvision_faster_rcnn_process import FasterRcnnFactory
        # Instantiate process object
        return FasterRcnnFactory()

    def getWidgetFactory(self):
        from infer_torchvision_faster_rcnn.infer_torchvision_faster_rcnn_widget import FasterRcnnWidgetFactory
        # Instantiate associated widget object
        return FasterRcnnWidgetFactory()

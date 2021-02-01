from ikomia import core, dataprocess
from ikomia.dnn.torch import models
import os
import copy
# Your imports below
import torch
import torchvision.transforms as transforms


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class FasterRCNNParam(core.CProtocolTaskParam):

    def __init__(self):
        core.CProtocolTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'FasterRcnn'
        self.dataset = 'Coco2017'
        self.model_path = ''
        self.classes_path = os.path.dirname(os.path.realpath(__file__)) + "/models/coco2017_classes.txt"
        self.confidence = 0.5
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.dataset = param_map["dataset"]
        self.model_path = param_map["model_path"]
        self.classes_path = param_map["classes_path"]
        self.confidence = float(param_map["confidence"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["model_name"] = self.model_name
        param_map["dataset"] = self.dataset
        param_map["model_path"] = self.model_path
        param_map["classes_path"] = self.classes_path
        param_map["confidence"] = str(self.confidence)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class FasterRCNNProcess(dataprocess.CImageProcess2d):

    def __init__(self, name, param):
        dataprocess.CImageProcess2d.__init__(self, name)
        self.model = None
        self.class_names = []
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Remove graphics input
        self.removeInput(1)
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CDblFeatureIO())

        # Create parameters class
        if param is None:
            self.setParam(FasterRCNNParam())
        else:
            self.setParam(copy.deepcopy(param))

    def load_class_names(self):
        self.class_names.clear()
        param = self.getParam()

        with open(param.classes_path) as f:
            for row in f:
                self.class_names.append(row[:-1])

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def predict(self, image):
        trs = transforms.Compose([
            transforms.ToTensor(),
            ])

        input_tensor = trs(image)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        return output

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        # Get input :
        input = self.getInput(0)
        src_image = input.getImage()

        h = src_image.shape[0]
        w = src_image.shape[1]

        # Step progress bar:
        self.emitStepProgress()

        # Load model
        if self.model is None or param.update:
            # Load class names
            self.load_class_names()
            # Load model
            use_torchvision = param.dataset != "Custom"
            self.model = models.faster_rcnn(use_pretrained=use_torchvision,
                                              classes=len(self.class_names))
            if param.dataset == "Custom":
                self.model.load_state_dict(torch.load(param.model_path))

            self.model.to(self.device)
            param.update = False

        pred = self.predict(src_image)
        cpu = torch.device("cpu")
        boxes = pred[0]["boxes"].to(cpu).numpy().tolist()
        scores = pred[0]["scores"].to(cpu).numpy().tolist()
        labels = pred[0]["labels"].to(cpu).numpy().tolist()

        # Step progress bar:
        self.emitStepProgress()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Set graphics output
        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("FasterRCNN")
        graphics_output.setImageIndex(0)

        detected_names = []
        detected_scores = []

        for i in range(len(boxes)):
            if scores[i] > param.confidence:
                # box
                box = boxes[i]
                w = box[2] - box[0]
                h = box[3] - box[1]
                graphics_output.addRectangle(float(box[0]), float(box[1]), float(w), float(h))
                # label
                label = self.class_names[labels[i]] + ": {:.3f}".format(scores[i])
                graphics_output.addText(label, float(box[0]), float(box[1]))
                detected_names.append(self.class_names[labels[i]])
                detected_scores.append(scores[i])

        # Init numeric output
        numeric_ouput = self.getOutput(2)
        numeric_ouput.clearData()
        numeric_ouput.setOutputType(dataprocess.NumericOutputType.TABLE)
        numeric_ouput.addValueList(detected_scores, "Probability", detected_names)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class FasterRCNNProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "FasterRCNN"
        self.info.shortDescription = "Faster R-CNN inference model for object detection."
        self.info.description = "Faster R-CNN inference model for object detection. " \
                                "Implementation from PyTorch torchvision package. " \
                                "This Ikomia plugin can make inference of pre-trained model from " \
                                "COCO dataset or custom trained model. Custom training can be made with " \
                                "the associated FasterRCNNTrain plugin from Ikomia marketplace."
        self.info.authors = "Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun"
        self.info.article = "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
        self.info.journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137-1149, 1 June 2017"
        self.info.year = 2017
        self.info.licence = "BSD-3-Clause License"
        self.info.documentationLink = "https://arxiv.org/abs/1506.01497"
        self.info.repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.iconPath = "icons/pytorch-logo.png"
        self.info.version = "1.0.0"
        self.info.keywords = "torchvision,detection,object,resnet,fpn,pytorch"

    def create(self, param=None):
        # Create process object
        return FasterRCNNProcess(self.info.name, param)

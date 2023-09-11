from ikomia import core, dataprocess
from ikomia.dnn.torch import models
import os
import copy
import random
import torch
import torchvision.transforms as transforms


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class FasterRcnnParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'FasterRcnn'
        self.dataset = 'Coco2017'
        self.model_weight_file = ''
        self.class_file = os.path.dirname(os.path.realpath(__file__)) + "/models/coco2017_classes.txt"
        self.conf_thres = 0.5
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.dataset = param_map["dataset"]
        self.model_weight_file = param_map["model_weight_file"]
        self.class_file = param_map["class_file"]
        self.conf_thres = float(param_map["conf_thres"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = self.model_name
        param_map["dataset"] = self.dataset
        param_map["model_weight_file"] = self.model_weight_file
        param_map["class_file"] = self.class_file
        param_map["conf_thres"] = str(self.conf_thres)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class FasterRcnn(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        self.model = None
        self.class_names = []
        self.colors = []
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create parameters class
        if param is None:
            self.set_param_object(FasterRcnnParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def load_class_names(self):
        self.class_names.clear()
        param = self.get_param_object()

        with open(param.class_file) as f:
            for row in f:
                self.class_names.append(row[:-1])

    def get_progress_steps(self):
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
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Temporary fix to clean detection outputs
        self.get_output(1).clear_data()

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        img_input = self.get_input(0)
        src_image = img_input.get_image()

        # Step progress bar:
        self.emit_step_progress()

        # Load model
        if self.model is None or param.update:
            # Load class names
            self.load_class_names()
            # Load model
            use_torchvision = param.dataset != "Custom"
            self.model = models.faster_rcnn(use_pretrained=use_torchvision, classes=len(self.class_names))
            if param.dataset == "Custom":
                self.model.load_state_dict(torch.load(param.model_weight_file, map_location=self.device))

            self.model.to(self.device)
            self.set_names(self.class_names)
            param.update = False

        pred = self.predict(src_image)
        cpu = torch.device("cpu")
        boxes = pred[0]["boxes"].to(cpu).numpy().tolist()
        scores = pred[0]["scores"].to(cpu).numpy().tolist()
        labels = pred[0]["labels"].to(cpu).numpy().tolist()

        # Step progress bar:
        self.emit_step_progress()

        for i in range(len(boxes)):
            if scores[i] > param.conf_thres:
                # box
                box_x = float(boxes[i][0])
                box_y = float(boxes[i][1])
                box_w = float(boxes[i][2] - boxes[i][0])
                box_h = float(boxes[i][3] - boxes[i][1])
                self.add_object(i, labels[i], float(scores[i]), box_x, box_y, box_w, box_h)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class FasterRcnnFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_torchvision_faster_rcnn"
        self.info.short_description = "Faster R-CNN inference model for object detection."
        self.info.description = "Faster R-CNN inference model for object detection. " \
                                "Implementation from PyTorch torchvision package. " \
                                "This Ikomia plugin can make inference of pre-trained model from " \
                                "COCO dataset or custom trained model. Custom training can be made with " \
                                "the associated FasterRCNNTrain plugin from Ikomia marketplace."
        self.info.authors = "Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun"
        self.info.article = "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
        self.info.journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137-1149, 1 June 2017"
        self.info.year = 2017
        self.info.license = "BSD-3-Clause License"
        self.info.documentation_link = "https://arxiv.org/abs/1506.01497"
        self.info.repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.icon_path = "icons/pytorch-logo.png"
        self.info.version = "1.3.0"
        self.info.keywords = "torchvision,detection,object,resnet,fpn,pytorch"

    def create(self, param=None):
        # Create process object
        return FasterRcnn(self.info.name, param)

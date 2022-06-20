import logging
import cv2
import numpy as np
from ikomia.core import task
from ikomia.utils.tests import run_for_test


logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer torchvision faster rcnn =====")
    logger.info("----- Use default parameters")
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[::-1]
    input_img_0 = t.getInput(0)
    input_img_0.setImage(img)
    return run_for_test(t)
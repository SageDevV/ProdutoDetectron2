import os

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo

from loss import ValidationLoss

import cv2


def get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes):
    """
    Create a Detectron2 configuration object and set its attributes.

    Args:
        output_dir (str): The path to the output directory where the trained model and logs will be saved.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size used during training.
        iterations (int): The maximum number of training iterations.
        checkpoint_period (int): The number of iterations between consecutive checkpoints.
        model (str): The name of the model to use, which should be one of the models available in Detectron2's model zoo.
        device (str): The device to use for training, which should be 'cpu' or 'cuda'.
        nmr_classes (int): The number of classes in the dataset.

    Returns:
        The Detectron2 configuration object.
    """
    cfg = _get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model))

    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("val",)
    cfg.DATASETS.TEST = ()

    if device in ['cpu']:
        cfg.MODEL.DEVICE = 'cpu'

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    cfg.SOLVER.IMS_PER_BATCH = batch_size

    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    cfg.SOLVER.BASE_LR = learning_rate

    cfg.SOLVER.MAX_ITER = iterations

    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes

    cfg.OUTPUT_DIR = output_dir

    return cfg


def get_dicts(img_dir, ann_dir):
    """
    Read the annotations for the dataset in YOLO format and create a list of dictionaries containing information for each
    image.

    Args:
        img_dir (str): Directory containing the images.
        ann_dir (str): Directory containing the annotations.

    Returns:
        list[dict]: A list of dictionaries containing information for each image. Each dictionary has the following keys:
            - file_name: The path to the image file.
            - image_id: The unique identifier for the image.
            - height: The height of the image in pixels.
            - width: The width of the image in pixels.
            - annotations: A list of dictionaries, one for each object in the image, containing the following keys:
                - bbox: A list of four integers [x0, y0, w, h] representing the bounding box of the object in the image,
                        where (x0, y0) is the top-left corner and (w, h) are the width and height of the bounding box,
                        respectively.
                - bbox_mode: A constant from the `BoxMode` class indicating the format of the bounding box coordinates
                             (e.g., `BoxMode.XYWH_ABS` for absolute coordinates in the format [x0, y0, w, h]).
                - category_id: The integer ID of the object's class.
    """
    dataset_dicts = []
    for idx, file in enumerate(os.listdir(ann_dir)):
        record = {}

        filename = os.path.join(img_dir, file[:-4] + '.jpg')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(os.path.join(ann_dir, file)) as r:
            lines = [l[:-1] for l in r.readlines()]

        for _, line in enumerate(lines):
            if len(line) > 2:
                label, cx, cy, w_, h_ = line.split(' ')

                obj = {
                    "bbox": [int((float(cx) - (float(w_) / 2)) * width),
                             int((float(cy) - (float(h_) / 2)) * height),
                             int(float(w_) * width),
                             int(float(h_) * height)],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": int(label),
                }

                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_datasets(root_dir, class_list_file):
    """
    Registers the train and validation datasets and returns the number of classes.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        class_list_file (str): Path to the file containing the list of class names.

    Returns:
        int: The number of classes in the dataset.
    """

    with open(class_list_file, 'r') as reader:
        classes_ = [l[:-1] for l in reader.readlines()]

    for d in ['train', 'val']:
        DatasetCatalog.register(d, lambda d=d: get_dicts(os.path.join(root_dir, d, 'imgs'),
                                                         os.path.join(root_dir, d, 'anns')))
        MetadataCatalog.get(d).set(thing_classes=classes_)

    return len(classes_)


def train(output_dir, data_dir, class_list_file, learning_rate, batch_size, iterations, checkpoint_period, device,
          model):
    """
    Train a Detectron2 model on a custom dataset.

    Args:
        output_dir (str): Path to the directory to save the trained model and output files.
        data_dir (str): Path to the directory containing the dataset.
        class_list_file (str): Path to the file containing the list of class names in the dataset.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        iterations (int): Maximum number of training iterations.
        checkpoint_period (int): Number of iterations after which to save a checkpoint of the model.
        device (str): Device to use for training (e.g., 'cpu' or 'cuda').
        model (str): Name of the model configuration to use. Must be a key in the Detectron2 model zoo.

    Returns:
        None
    """

    nmr_classes = register_datasets(data_dir, class_list_file)

    cfg = get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)

    val_loss = ValidationLoss(cfg)

    trainer.register_hooks([val_loss])

    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    trainer.resume_or_load(resume=False)

    trainer.train()

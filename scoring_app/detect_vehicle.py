# Original source: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/neural_style.py
import argparse
import os
import time
import sys
import re
import logging
import util

from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms

import visualize


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize(
            (int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS
        )
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def _detect(content_scale, detection_model, device, input_file, output_file, output_dir):
    """
    :param content_scale: to scale image
    :param detection_model: the detection model
    :param device: cuda or cpu
    :param path: full path of image to process
    :param filename: the name of the file to output
    :param output_dir: the name of the dir to save processed output files
    """
    
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
    
    logger = logging.getLogger("root")

    logger.debug("Processing {}".format(input_file))
    content_image = load_image(input_file, scale=content_scale)
    content_transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    image_tensor = content_transform(content_image)
    image_tensor = [image_tensor.to(device)]
    output_path = os.path.join(output_dir, output_file)

    with torch.no_grad():
        results = detection_model(image_tensor)

    if len(results[0]["scores"]) > 0:
        # Result selection:
        #   Label in ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
        #   Score > 0.5
        label_selection = [2, 3, 4, 6, 8]
        score_thres = 0.5
        selected_results = []
        for i, result in enumerate(results):
            select_mask = torch.tensor([True if l in label_selection else False for l in result["labels"]])
            select_mask &= torch.tensor([True if s > score_thres else False for s in result["scores"]])
            selected_results.append({k:v[select_mask].detach().cpu().numpy() for k, v in result.items()})
        r = selected_results[0]
        
        visualize.display_instances(
            np.array(content_image),
            np.array([[b[1], b[0], b[3], b[2]] for b in r['boxes'].astype(int)]),
            r['masks'].transpose((1,2,3,0)).squeeze(0),
            r['labels'],
            class_names,
            r['scores'],
            show_image=False,
            save_to=output_path)
    else:
        content_image.save(output_path)


def detect(content_scale, content_filename, model_dir, cuda, content_dir, output_dir):

    logger = logging.getLogger("root")

    # check that all the paths and image references are good
    assert os.path.exists(content_dir)
    assert os.path.exists(output_dir)
    if model_dir:
        assert os.path.isdir(model_dir)
    if content_filename:
        assert os.path.exists(os.path.join(content_dir, content_filename))

    device = torch.device("cuda" if cuda else "cpu")
    with torch.no_grad():
        if model_dir and os.path.isfile(model_dir):
            #logger.debug("Use the model '"+model_dir+"'.")
            detection_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
            state_dict = torch.load(os.path.join(model_dir, "model.pth"))
            detection_model.load_state_dict(state_dict)
        else:
            #logger.debug("Cannot find '"+model_dir+"'. Use the pretrained model.")
            detection_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        detection_model.eval()
        detection_model.to(device)

        # if applying style transfer to only one image
        if content_filename:
            full_path = os.path.join(content_dir, content_filename)
            _detect(
                content_scale,
                detection_model,
                device,
                full_path,
                content_filename,
                output_dir,
            )

        # if applying style transfer to all images in directory
        else:
            filenames = os.listdir(content_dir)
            for filename in filenames:
                full_path = os.path.join(content_dir, filename)
                _detect(
                    content_scale, detection_model, device, full_path, filename, output_dir
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    parser.add_argument(
        "--content-scale",
        type=float,
        default=None,
        help="factor for scaling down the content image",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=False,
        default="",
        help="saved model dir the contains model.pth",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="set it to 1 for running on GPU, 0 for CPU",
    )
    parser.add_argument(
        "--content-dir", type=str, required=True, help="directory holding the images"
    )
    parser.add_argument(
        "--content-filename",
        type=str,
        help="(optional) if specified, only process the individual image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="directory holding the output images",
    )
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    # set up logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(util.get_handler_format())
    logger = logging.getLogger("root")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.debug("Starting vehicle detection...")
    detect(
        content_scale=args.content_scale,
        model_dir=args.model_dir,
        cuda=args.cuda,
        content_dir=args.content_dir,
        content_filename=args.content_filename,
        output_dir=args.output_dir,
    )

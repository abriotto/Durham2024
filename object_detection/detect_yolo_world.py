import os
import json
import argparse
from ultralytics import YOLOWorld

def get_params(params):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cpu',
                        help="Specifies the device for tensor computations. Defaults to 'cpu'.")
    parser.add_argument("--model", default='YOLOWorld', type=str)
    parser.add_argument("--dataset", default='brueg_small', type=str)
    parser.add_argument("--save", default=True, type=bool)
    parser.add_argument("--conf", default=0.1, type=float)
    return parser.parse_args(params)

def detect(opts):
    if opts.model == 'YOLOWorld':
        model = YOLOWorld("object_detection/yolov8x-worldv2.pt")
        model.set_classes(['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'mouse', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'mountain', 'tree', 'house', 'boat', 'mill'])
    else:
        raise NotImplementedError

    dataset_folder = 'datasets/' + opts.dataset
    detection_folder = 'object_detection/' + opts.dataset + '_detections'

    os.makedirs(detection_folder, exist_ok=True)

    for img_name in os.listdir(dataset_folder):
        img_path = os.path.join(dataset_folder, img_name)
        print(opts.device)
        results = model.predict(img_path, conf=opts.conf, device=opts.device, half=True)
        results_json = results.tojson(normalize=True, decimals=3)  # Convert results to JSON

        # Generate JSON file name based on image name
        json_filename = os.path.splitext(img_name)[0] + '.json'
        json_filepath = os.path.join(detection_folder, json_filename)

        with open(json_filepath, 'w') as json_file:
            json_file.write(results_json)  # Write JSON to file
            print('saved to json!')

def main(params):
    opts = get_params(params)
    detect(opts)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

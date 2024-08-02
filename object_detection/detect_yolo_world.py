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
    parser.add_argument("--iou_thres", default=0.5, type=float, help="IOU threshold for NMS")
    parser.add_argument("--max_detections", default = 100 )
    return parser.parse_args(params)


def load_classes(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Flatten the list of lists into a single list of strings ##ASK IF MAKES SENSE
    classes = [item for sublist in data for item in sublist]
    return classes


def detect(opts):
    if opts.model == 'YOLOWorld':
        model = YOLOWorld("object_detection/yolov8x-worldv2.pt")
        classes = load_classes('object_detection/lvis_v1_class_texts.json')
        print(classes[:10])  # Print the first 10 classes to verify 

        model.set_classes(classes)
      


    dataset_folder = 'datasets/' + opts.dataset
    detection_folder = 'object_detection/' + opts.dataset + '_detections'

    os.makedirs(detection_folder, exist_ok=True)

    for img_name in os.listdir(dataset_folder):
        img_path = os.path.join(dataset_folder, img_name)
        print(opts.device)
        results = model.predict(img_path, conf=opts.conf, device=opts.device, half=True, iou=opts.iou_thres, agnostic_nms = True, max_det = 100)
        results_json = results[0].tojson(normalize=True, decimals=3)  # Convert results to JSON

        # Generate JSON file name based on image name
        json_filename = os.path.splitext(img_name)[0] + '.json'
        json_filepath = os.path.join(detection_folder, json_filename)

        with open(json_filepath, 'w') as json_file:
            json_file.write(results_json)  # Write JSON to file
            print('Saved to JSON:', json_filepath)

def main(params):
    opts = get_params(params)
    detect(opts)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

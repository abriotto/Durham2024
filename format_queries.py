import json
import ujson
import os
import argparse
import utils

def main(query_file_path, query_detection_folder, img_folder):
    cats_dict = {}
    cats = []
    file_names = []

    if not os.path.exists(query_detection_folder):
        os.mkdir(query_detection_folder)
    
    with open(query_file_path, 'r') as query_file:
        queries = ujson.load(query_file)

    for category, items in queries.items():
        for item in items:
             # Generate JSON file name based on image name
            json_filename = (item['query'][0]).replace('.jpg', '.json')
            json_filepath = os.path.join(query_detection_folder, json_filename)
            
            #patch dict is saved wrapped in list to maintain same structre as in yolo detections
            #retrieve bounding box

            box = item['query'][1]
            query_path = os.path.join(img_folder, item['query'][0])
            box_ratios = utils.box_pixels_to_ratio(box, query_path)
            global_matches = [match[0] for match in item['gt']]

            p_d = [
                {'box':{
                 'x1': box_ratios[0],
                 'y1': box_ratios[1],
                 'x2': box_ratios[2],
                 'y2': box_ratios[3]},
                 
                 'category': category,
                 
                 'global_matches': global_matches}
            ]

            if category not in cats:
                cats.append(category)
                cats_dict[category] = global_matches
             


            #TO FIX!!!!!!! IN THIS WAY ONLY ONE QUERY IS FOUND
            
            if json_filename in file_names:
                with open(json_filepath, 'r') as f:
                    dict_list= json.load(f)
                dict_list.extend(p_d)
                with open(json_filepath, 'w') as f:
                    json.dump(dict_list, f, indent = 2)



            else:
                with open(json_filepath, 'w') as f:
                    json.dump(p_d, f, indent=2)
            
            file_names.append(json_filename)
            cats.append(category)
            
    with open(os.path.basename(query_file_path)+'_categories.json', 'w') as f:
        json.dump(cats_dict, f, indent=2)
    
    ##make the dictionaries in one list
        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file_path', type=str)
    parser.add_argument('--query_detection_folder', type=str)
    parser.add_argument('--img_folder', type=str)


    args = parser.parse_args()
    main(args.query_file_path, args.query_detection_folder, args.img_folder)
    


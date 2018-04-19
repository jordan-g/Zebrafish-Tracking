import sys
import argparse
import json
import numpy as np
import tracking

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="Path to the video you want to track.", required=True)
parser.add_argument("-b", "--background", help="Optional argument to include path to background image for background subtraction.", default=None)
parser.add_argument("-p", "--params", help="Path to the tracking parameters JSON file.", required=True)
parser.add_argument("-o", "--output", help="Path to where to save tracking output.", required=True)
args = parser.parse_args()

# extract video path, tracking params and tracking path
video_path = args.video
background_path = args.background
params_json_string = open(args.params).read()
params = json.loads(params_json_string)
tracking_path = args.output

# fix up the params dictionary (JSON doesn't allow encoding/decoding numpy arrays)
# convert offset lists to numpy arrays
for i in range(len(params['crop_params'])):
    params['crop_params'][i]['offset'] = np.array(params['crop_params'][i]['offset'])
# convert body crop list to numpy array
params['body_crop'] = np.array(params['body_crop'])

# fill in a missing parameter
params['backgrounds'] = None
# params['background_path'] = None
# params['background_path'] = "C:\\Users\\Nicholas\\Desktop\\LOOM-1_Camera_Video_background.tif"

# track the video
tracking.open_and_track_video(video_path, background_path, params, tracking_path)

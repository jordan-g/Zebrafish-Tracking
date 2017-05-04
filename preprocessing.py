import numpy as np
import cv2

# --- Background subtraction --- #

def calc_background(frames):
    background = np.amax(frames, axis=0)
    return background

def subtract_background_from_frame(frame, background, rescale_brightness=True):
    # subtract background from frame
    bg_sub_frame = frame - background
    bg_sub_frame[bg_sub_frame < 0] = 255

    if rescale_brightness:
        bg_sub_frame = rescale_dynamic_range(bg_sub_frame)

    return bg_sub_frame

def subtract_background_from_frames(frames, background, rescale_brightness=True):
    # initialize array of background subtracted frames
    bg_sub_frames = np.zeros(frames.shape)

    bg_sub_frames = frames - background
    bg_sub_frames[bg_sub_frames < 0] = 255

    if rescale_brightness:
        bg_sub_frames = rescale_dynamic_range(bg_sub_frames)

    return bg_sub_frames

# --- Dynamic range / threshold calculation --- #

def rescale_dynamic_range(frames):
    rescaled_frames = []

    max_brightness = max([ np.amax(frame) for frame in frames ])
    min_brightness = min([ np.amin(frame) for frame in frames ])

    for frame in frames:
        rescaled_frames.append(((frame - min_brightness)*255.0/(max_brightness - min_brightness)).astype(np.uint8))

    return rescaled_frames
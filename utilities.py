import numpy as np
from scipy.ndimage import zoom

try:
    xrange
except:
    xrange = range

def clipped_zoom(img, zoom_factor, **kwargs):
    '''
    Efficiently zoom an image , preserving aspect ratio, by clipping it before zooming.
    Obtained from https://stackoverflow.com/a/37121993

    Arguments:
        img (ndarray)       : 2D image to zoom.
        zoom_factor (float) : Factor by which to zoom the image.
        kwargs              : Extra arguments to the scipy.ndimage.zoom function.

    Returns:
        out (ndarray) : Zoomed image.
    '''

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def split_evenly(n, m, start=0):
    '''
    Return a list of m evenly-spaced integers in the range (start, start + n).
    For example, split_evenly(100, 5, 30) = [40, 60, 80, 100, 120]

    Arguments:
        n (int)     : Range size.
        m (int)     : Number of integers to return.
        start (int) : Starting integer of range.

    Returns:
        l (list) : List of evenly-spaced integers from the provided range.
    '''

    return [i*n//m + n//(2*m) + start for i in range(m)]

def yield_chunks_from_array(array, n):
    '''
    Yield successive n-sized chunks from an array.

    Arguments:
        array (ndarray) : Input array.
        n (int)         : Size of each chunk to yield.

    Yields:
        chunk (ndarray) : n-sized chunk from the input array.
    '''

    for i in xrange(0, array.shape[0], n):
        yield array[i:i + n]

def split_list_into_chunks(l, n):
    '''
    Return a list of n-sized chunks from l.

    Arguments:
        l (list) : Input list.
        n (int)  : Size of each chunk.

    Returns:
        l_2 (list) : List of n-sized chunks of l.
    '''

    return [ l[i:i + n] for i in xrange(0, len(l), n) ]
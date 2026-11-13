"""
make_tiff.py -- one-time conversion of the benchmark AVI to a multi-page TIFF
for MIN1PIPE, with a frame-by-frame check that no pixel changed.

  python -m bench.make_tiff C:/.../data/synthetic_5min.avi C:/.../data/min1pipe_input/synthetic_5min.tif

Why MIN1PIPE needs this: its AVI reader (utilities/elements/data_cat.m) does
not decode video. It memory-maps the file and pulls raw frame bytes out of the
AVI container, which only works for uncompressed AVIs from the UCLA Miniscope
software, and it calls aviinfo(), which current MATLAB releases no longer
have. TIFF input goes through imfinfo/imread and avoids both problems.

The conversion happens once, before any timed run, and is not counted in
MIN1PIPE's time. State this in the methods: every tool got identical pixels,
each in a format it reads.

Put the TIFF in its own folder (not next to the AVI): MPS loads every .avi in
the video folder, and MIN1PIPE looks for files by name prefix.
"""

import argparse
import os
import sys

import numpy as np


def read_frames(path):
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError("cv2 could not open %s" % path)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr[..., 0] if fr.ndim == 3 else fr
        cap.release()
    except ImportError:
        import imageio
        for fr in imageio.get_reader(path, "ffmpeg"):
            yield fr[..., 0] if fr.ndim == 3 else fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("avi")
    ap.add_argument("tif")
    a = ap.parse_args()
    import tifffile

    os.makedirs(os.path.dirname(os.path.abspath(a.tif)), exist_ok=True)
    n = 0
    with tifffile.TiffWriter(a.tif, bigtiff=False) as tw:
        for fr in read_frames(a.avi):
            tw.write(np.ascontiguousarray(fr.astype(np.uint8)), contiguous=True)
            n += 1
    print("wrote %d frames -> %s" % (n, a.tif))

    # verify: decode the AVI again and compare every frame with the TIFF
    with tifffile.TiffFile(a.tif) as tf:
        if len(tf.pages) != n:
            sys.exit("page count %d != frame count %d" % (len(tf.pages), n))
        for i, fr in enumerate(read_frames(a.avi)):
            if not np.array_equal(tf.pages[i].asarray(), fr.astype(np.uint8)):
                sys.exit("frame %d differs between AVI and TIFF" % i)
    print("verified: all %d frames identical" % n)


if __name__ == "__main__":
    main()

"""
check_video.py -- confirm the synthetic AVI is what the benchmark assumes
before spending hours on runs.

  python -m bench.check_video D:\\bench\\data\\synth.avi --gt-frames 3000

Reports codec, frame count, fps, size, and channels, and checks that the
frames decode identically on two passes. Warns when the codec is lossy: if
the AVI was compressed with MJPG/XVID/H.264, the pixels the pipelines see are
not the pixels the ground truth was generated from.
"""

import argparse
import sys

import cv2
import numpy as np

LOSSLESS = {"FFV1", "HFYU", "MJ2C", "LAGS", "Y800", "GREY", "RAW ", "DIB ", "\x00\x00\x00\x00",
            "Y8  ", "I420"}
LOSSY = {"MJPG", "XVID", "DIVX", "H264", "AVC1", "X264", "MP4V", "MPEG", "FMP4", "HEVC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gt-frames", type=int, help="number of frames in the ground truth")
    ap.add_argument("--dims", type=int, nargs=2, metavar=("H", "W"))
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit("could not open %s" % a.video)
    fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_i >> 8 * i) & 0xFF) for i in range(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    first = None
    color = False
    mx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if first is None:
            first = fr.copy()
            color = fr.ndim == 3 and not (np.array_equal(fr[..., 0], fr[..., 1]) and
                                          np.array_equal(fr[..., 1], fr[..., 2]))
        mx = max(mx, int(fr.max()))
        n += 1
    cap.release()
    h, w = first.shape[:2]
    print("file       : %s" % a.video)
    print("codec      : %r" % fourcc)
    print("fps        : %.3f" % fps)
    print("frames     : %d decoded (%d in header)" % (n, n_meta))
    print("size       : %d x %d (H x W)" % (h, w))
    print("channels   : %s" % ("colour" if color else "grey (stored as %d ch)" % (first.shape[2] if first.ndim == 3 else 1)))
    print("max value  : %d" % mx)
    print("duration   : %.1f s" % (n / fps if fps else float("nan")))

    problems = []
    fc = fourcc.upper()
    if fc in LOSSY:
        problems.append("codec %s is LOSSY: re-export the synthetic video with FFV1 or uncompressed "
                        "so the pipelines see the same pixels the ground truth describes" % fourcc)
    elif fc not in LOSSLESS:
        problems.append("codec %r not recognised; confirm with whoever generated the data that it is lossless" % fourcc)
    if color:
        problems.append("frames are colour; the pipelines expect greyscale")
    if a.gt_frames and a.gt_frames != n:
        problems.append("video has %d frames but ground truth has %d" % (n, a.gt_frames))
    if a.dims and tuple(a.dims) != (h, w):
        problems.append("size %dx%d differs from expected %dx%d" % (h, w, *a.dims))
    if n != n_meta:
        problems.append("decoded frame count differs from header; some readers may disagree on length")
    print()
    if problems:
        for p in problems:
            print("WARNING:", p)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()

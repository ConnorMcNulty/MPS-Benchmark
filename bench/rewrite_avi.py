"""
rewrite_avi.py -- rebuild the benchmark AVI from the verified TIFF so that the
frame count stored in the AVI header matches the frames actually in the file.

  python -m bench.rewrite_avi ../min1pipe_input/synthetic_5min.tif ../data/synthetic_5min.avi

Why: the original AVI's header says 3053 frames but only 3000 decode. Readers
that decode front to back (MPS, OpenCV) are unaffected, but minian's loader
takes the frame count from the header (ffprobe nb_frames) and reshapes the
decoded bytes to that length, and CaImAn pre-allocates from
CAP_PROP_FRAME_COUNT. A clean header avoids both problems. The TIFF is the
source because bench.make_tiff and bench.check_alignment already showed it is
pixel-identical to the decoded AVI and lined up with the ground truth.

Output codec is FFV1 (lossless). Every frame is checked after writing, and the
header is checked with OpenCV and, if ffprobe/ffmpeg are on PATH, with those too.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("avi_out")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()
    import cv2
    import tifffile

    with tifffile.TiffFile(a.tif) as tf:
        V = np.stack([p.asarray() for p in tf.pages]).astype(np.uint8)
    n, h, w = V.shape
    if os.path.exists(a.avi_out):
        sys.exit("%s exists; move the original out of the data folder first" % a.avi_out)

    vw = cv2.VideoWriter(a.avi_out, cv2.VideoWriter_fourcc(*"FFV1"), a.fps, (w, h), isColor=False)
    if not vw.isOpened():
        sys.exit("OpenCV could not open an FFV1 writer")
    for fr in V:
        vw.write(fr)
    vw.release()
    print("wrote %d frames (%dx%d, %.1f fps) -> %s" % (n, h, w, a.fps, a.avi_out))

    problems = []
    cap = cv2.VideoCapture(a.avi_out)
    hdr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = fr[..., 0] if fr.ndim == 3 else fr
        if i >= n or not np.array_equal(g, V[i]):
            problems.append("frame %d differs from the TIFF" % i)
            break
        i += 1
    cap.release()
    print("OpenCV: header %d frames, decoded %d frames" % (hdr, i))
    if hdr != n:
        problems.append("OpenCV header count %d != %d" % (hdr, n))
    if i != n:
        problems.append("OpenCV decoded %d != %d" % (i, n))

    if shutil.which("ffprobe"):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_frames", "-of", "json", a.avi_out],
                             capture_output=True, text=True).stdout
        nb = int(json.loads(out)["streams"][0]["nb_frames"])
        print("ffprobe: nb_frames %d" % nb)
        if nb != n:
            problems.append("ffprobe nb_frames %d != %d" % (nb, n))
    else:
        print("ffprobe not on PATH in this environment (skipped)")
    if shutil.which("ffmpeg"):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.avi_out, "-f", "rawvideo",
                              "-pix_fmt", "gray", "pipe:"], capture_output=True).stdout
        nf = len(raw) / (h * w)
        print("ffmpeg rawvideo decode: %g frames" % nf)
        if nf != n:
            problems.append("ffmpeg decoded %g frames != %d" % (nf, n))
        elif not np.array_equal(np.frombuffer(raw, np.uint8).reshape(n, h, w), V):
            problems.append("ffmpeg-decoded pixels differ from the TIFF")
    else:
        print("ffmpeg not on PATH in this environment (skipped)")

    if problems:
        print("\nPROBLEM:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nOK: header and decoded frame count both %d; pixels identical to the TIFF" % n)


if __name__ == "__main__":
    main()

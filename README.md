# yolov9-sahi

YOLOv9 inference using [ncnn](https://github.com/Tencent/ncnn) for a full image utilizing the
SAHI method (Slicing Aided Hyper Inference): the image is cut into overlapping tiles at
the model's input size, each tile is run through YOLOv9, and detections are mapped back
and merged across tiles. Small objects in large images stay at native resolution.

## Layout

```
yolov9-sahi/
├── CMakeLists.txt
├── main.cpp
├── ncnn/                     # git submodule (Tencent/ncnn, release 20261020)
└── models/
    └── LibreYOLO9t_ncnn/     # not committed: model.ncnn.param, model.ncnn.bin, metadata.yaml
```

## Clone

```bash
git clone --recursive https://github.com/detu/yolov9-sahi
# already cloned without --recursive:
git submodule update --init --recursive
```

## Model

The model comes from [LibreYOLO](https://github.com/LibreYOLO/libreyolo) (MIT license),
exported to ncnn (any machine with Python; WSL is fine):

```bash
pip install "libreyolo[ncnn]"
libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640   # or your own trained checkpoint
mv weights/LibreYOLO9t_ncnn models/
```

This writes `model.ncnn.param`, `model.ncnn.bin` and `metadata.yaml`. Pass the two model
files; the `metadata.yaml` next to the `.param` file is read automatically for the class
names, input size, box format and letterbox placement:

```bash
./build/yolov9_sahi models/LibreYOLO9t_ncnn/model.ncnn.param models/LibreYOLO9t_ncnn/model.ncnn.bin images/
```

LibreYOLO's YOLOv9 (including the NMS-free E2E variant) outputs one `(4 + classes) × anchors`
tensor: `x1, y1, x2, y2` in input pixels, then sigmoid class scores. LibreYOLO-trained
checkpoints letterbox the image into the top-left corner; converted official checkpoints
centre it; `metadata.yaml` records which (`letterbox_pad`).

Without a `metadata.yaml`, the box format (`x1,y1,x2,y2` or centre/width/height) is
detected from the model's output, the image is centred, and class names come from
`--names`. This also covers other YOLOv9 ncnn conversions, e.g. the YOLOv9-C model from
[twlelev/Yolov9-Ncnn](https://github.com/twlelev/Yolov9-Ncnn) (blobs `images` → `output0`,
picked up automatically). `--box`, `--pad`, `--in` and `--out` override the detection.

## Build

Requires CMake ≥ 3.18, a C++17 compiler with OpenMP, and OpenCV (core, imgproc,
imgcodecs, dnn). ncnn is built from the submodule; the build type defaults to Release.

```bash
cmake -B build
cmake --build build -j 4
```

Options:

```bash
cmake -B build -DSAHI_VULKAN=ON        # ncnn with Vulkan GPU support (run with --gpu)
cmake -B build -DSAHI_SYSTEM_NCNN=ON -Dncnn_DIR=<prefix>/lib/cmake/ncnn   # use an installed ncnn
```

ncnn runs fast on CPU, so the default build is CPU-only. **WSL2:** Vulkan in WSL2 goes
through Mesa's Dozen driver (Vulkan on Direct3D 12), which is very slow, so stay on CPU
there, or build natively on Windows/Linux for `--gpu`.

## Run

Arguments are, in order: the model's `.param` file, its `.bin` file, then one or more
images and/or directories:

```
yolov9_sahi <model.param> <model.bin> <image|dir> [<image|dir> ...] [options]
```

```bash
# one image
./build/yolov9_sahi models/yolov9.param models/yolov9.bin image.jpg -o result.jpg

# a whole folder (add -r for subfolders), with a CSV of all detections
./build/yolov9_sahi models/yolov9.param models/yolov9.bin images/ -o results/ --csv detections.csv
```

| Option | Default | Meaning |
|---|---|---|
| `-o, --output` | `sahi_output.jpg` / `sahi_output/` | output file for one image, output directory for several (`<name>_sahi.<ext>`) |
| `--csv` | off | write all detections: `image,class_id,class_name,score,x1,y1,x2,y2` |
| `-n, --names` | `metadata.yaml` | text file of class names, one per line: line 1 = class 0, line 2 = class 1, ... |
| `-r, --recursive` | off | scan directories recursively |
| `-s, --size` | `metadata.yaml` imgsz, else 640 | model input size |
| `--slice` | input size | tile size (tiles are letterboxed to the input size) |
| `--full` | off | also detect on the whole image (catches objects larger than a tile) |
| `-w, --workers` | 2 | tiles processed in parallel (OpenMP) |
| `-t, --threads` | cores / workers | ncnn threads per worker |
| `--gpu` | off | Vulkan (needs `-DSAHI_VULKAN=ON`) |
| `--overlap` | 0.20 | tile overlap ratio |
| `--conf` | 0.25 | confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold within a tile |
| `--merge` | `nmm` | merging across tiles: `nmm` merges overlapping boxes into one, `nms` keeps the best |
| `--match` | `ios` | overlap metric for the merge: `ios` (intersection over smaller box) or `iou` |
| `--match-thresh` | 0.5 | overlap above which boxes are merged |
| `--box` | `metadata.yaml`, else detected | box format in the output: `xyxy` or `cxcywh` |
| `--pad` | `metadata.yaml`, else `center` | letterbox placement: `center` or `topleft` |
| `--in`, `--out` | model's first input/output (`in0`, `out0`) | ncnn blob names |

Each tile is letterboxed (gray padding, centred or top-left) to the input size, normalized to 0–1 RGB
and run through the network; boxes are decoded (best class per anchor), mapped back to
the image and de-duplicated within the tile with class-aware NMS.

Across tiles, an object cut by a tile edge appears as a full box from one tile plus a
fragment from its neighbour, or as two halves. As in the SAHI library, the default merge
is greedy NMM with the IOS metric: boxes of the same class whose intersection covers more
than half of the smaller box are merged into their union, keeping the best score. Use
`--merge nms --match iou` for plain NMS. SAHI works best when objects are smaller than a
tile; add `--full` when the images also contain objects much larger than a tile. All workers share
one copy of the model (ncnn runs several extractors on one network concurrently), and
the next image is decoded in the background while the current one is processed. Output
is deterministic regardless of the number of workers.

On CPU, keep `workers × threads` at or below your physical core count; many workers with
few threads each usually gives the best throughput on many tiles.

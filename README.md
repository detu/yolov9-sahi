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
    └── yolov9c_ncnn_model/   # not committed: model.ncnn.param, model.ncnn.bin, metadata.yaml
```

## Clone

```bash
git clone --recursive https://github.com/detu/yolov9-sahi
# already cloned without --recursive:
git submodule update --init --recursive
```

## Model

Export a YOLOv9 model to ncnn with [Ultralytics](https://docs.ultralytics.com/integrations/ncnn/)
(any machine with Python; WSL is fine):

```bash
pip install ultralytics
yolo export model=yolov9c.pt format=ncnn imgsz=640     # or your own trained best.pt
mv yolov9c_ncnn_model models/
```

This writes `model.ncnn.param`, `model.ncnn.bin` and `metadata.yaml`. Pass the folder to
the program: the class names and input size are read from `metadata.yaml`.

Models converted another way work too, e.g. the YOLOv9-C model from
[twlelev/Yolov9-Ncnn](https://github.com/twlelev/Yolov9-Ncnn) (ONNX → ncnn, blobs
`images` → `output0`; names come from `--names`, and the blob names are picked up
automatically). The requirement is the usual
`(4 + classes) × anchors` tensor (box centre x/y, width, height in input pixels, then the
sigmoid class scores). Pass the `.param` file, and `--in`/`--out` if the blob names aren't
the first input/output.

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

The model is a `.param` + `.bin` pair. Pass the `.param` file (the `.bin` with the same
name next to it is used), both files, or an Ultralytics export folder:

```bash
./build/yolov9_sahi models/yolov9.param image.jpg                       # uses models/yolov9.bin
./build/yolov9_sahi models/yolov9.param models/weights.bin image.jpg    # .bin named differently
./build/yolov9_sahi models/yolov9c_ncnn_model image.jpg                 # Ultralytics export folder
```

```bash
# one image
./build/yolov9_sahi models/yolov9c_ncnn_model image.jpg -o result.jpg

# a whole folder (add -r for subfolders), with a CSV of all detections
./build/yolov9_sahi models/yolov9c_ncnn_model images/ -o results/ --csv detections.csv
```

| Option | Default | Meaning |
|---|---|---|
| `-b, --bin` | `.param` name with `.bin` | model weights, if not named like the `.param` (or give it as the 2nd argument) |
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
| `--in`, `--out` | model's first input/output (`in0`, `out0`) | ncnn blob names |

Each tile is letterboxed (gray padding, centred) to the input size, normalized to 0–1 RGB
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

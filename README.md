# yolov9-sahi

YOLOv9 inference using [ncnn](https://github.com/Tencent/ncnn) for a full image using
tiling: the image is cut into model-sized tiles overlapping by 50%, each tile is run
through YOLOv9 at native resolution, detections cut by an interior tile edge are dropped
(the neighbouring tile sees those objects whole), and the rest are merged with one NMS.
Small objects in large images stay at native resolution.

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
| `-o, --output` | `output.jpg` / `output/` | output file for one image, output directory for several (`<name>_det.<ext>`) |
| `--csv` | off | write all detections: `image,class_id,class_name,score,x1,y1,x2,y2` |
| `-n, --names` | `metadata.yaml` | text file of class names, one per line: line 1 = class 0, line 2 = class 1, ... |
| `-r, --recursive` | off | scan directories recursively |
| `-s, --size` | `metadata.yaml` imgsz, else 640 | model input size = tile size |
| `--stride` | half a tile | tile stride in pixels (default: 50% overlap) |
| `--margin` | 8 | drop detections within this many pixels of an interior tile edge |
| `--pre-scale` | 1 | downscale the image by this factor (0–1] before tiling: fewer tiles, larger objects fit |
| `-w, --workers` | physical cores | tiles run in parallel (OpenMP) |
| `-t, --threads` | cores / workers | ncnn threads per tile |
| `-j, --cores` | physical cores | total thread budget |
| `--gpu` | off | Vulkan (needs `-DSAHI_VULKAN=ON`) |
| `--conf` | 0.25 | confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold, within a tile and for the final merge |
| `--box` | `metadata.yaml`, else detected | box format in the output: `xyxy` or `cxcywh` |
| `--pad` | `metadata.yaml`, else `center` | letterbox placement: `center` or `topleft` |
| `--in`, `--out` | model's first input/output (`in0`, `out0`) | ncnn blob names |

### How the tiling works

1. **Work image.** With `--pre-scale f < 1` the image is first downscaled (area
   averaging). If it is smaller than one tile, it is padded at the bottom/right with gray
   (114) to one tile, so coordinates don't move.
2. **Tile grid.** Tiles are `size × size` (the model input size) at every `stride` pixels
   (default half a tile, i.e. 50% overlap). The number of tiles per row/column is the
   smallest that covers the image, and the last tile is clamped to the image border, so
   there are no partial or duplicate tiles. Tiles are read in place, not copied.
3. **Per tile.** The tile is normalized to 0–1 RGB and run through YOLOv9; boxes are
   decoded (best class per anchor) and de-duplicated with class-aware NMS.
4. **Edge filter.** A detection within `--margin` pixels of an *interior* tile edge (an
   edge that isn't the image border) is dropped: it is probably cut, and with 50% overlap
   the neighbouring tile sees the object whole. Detections on the real image border are
   kept.
5. **Merge.** The rest is mapped back to the original image (undoing the pre-scale),
   clipped to the real image (dropping the gray padding) and merged with one class-aware
   NMS, which removes objects seen whole by two overlapping tiles.

Output is deterministic regardless of the number of threads.

Objects larger than about a tile can't be seen whole by any tile, so the edge filter
drops them. Use `--pre-scale` to shrink such images until the large objects fit in a
tile. For example, `--pre-scale 0.5` turns a 4000 px wide image into tiles that cover
1280 px of the original.

### Threading

By default the program runs **one tile per physical core, with ncnn single-threaded**.
Tiling produces many equal tiles, and running them side by side scales almost linearly,
while ncnn's own per-layer threading scales worse on a single 640×640 tile and nesting the
two levels costs extra. Measured on 2 cores with YOLOv9-C, 35 tiles:

| Setup | Inference | vs 1 core |
|---|---|---|
| 1 core | 32.0 s | 1.0× |
| 1 tile × 2 ncnn threads | 21.8 s | 1.47× |
| 2 tiles × 1 thread (default) | 16.1 s | 1.99× |

Details:

- The split is fixed at startup: ncnn's convolution layers fix their thread count when
  the model is loaded. `-w` (tiles in parallel) and `-t` (ncnn threads per tile) override
  it; `-j` sets the total budget (default: physical cores; hyperthreads add little here).
- Images that fit in a single tile can't use tile parallelism. For those, a second copy
  of the model with all cores is loaded the first time such an image appears.
- All tiles share one copy of the model weights; each worker has its own ncnn memory pools
  (by default ncnn's extractors share one locked pool, which serializes concurrent tiles).
- With several tiles in flight, ncnn threads don't spin-wait between layers, so idle
  threads don't take cores from other tiles.
- The next image is decoded and the previous result is drawn and saved in the background
  while the current image runs.

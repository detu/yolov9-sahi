# yolov9-sahi

YOLOv9 inference using [ncnn](https://github.com/Tencent/ncnn) or ONNX Runtime with CUDA
for a full image utilizing the SAHI method (Slicing Aided Hyper Inference): the image is
cut into overlapping slices at the model's input size, each slice is letterboxed and run
through YOLOv9, and detections are mapped back and merged across slices as in the SAHI
library. Small objects in large images stay at native resolution.

This is the `sahi-cuda` branch: SAHI with the CUDA backend. The `cuda-backend` branch
has the same backend with tiling instead of SAHI; `main` has tiling with ncnn.

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
imgcodecs, dnn). The build type defaults to Release. Pick the inference backend with
`SAHI_BACKEND`:

| `SAHI_BACKEND` | Engine | Model | Runs on |
|---|---|---|---|
| `ORT_CUDA` (default on this branch) | ONNX Runtime, CUDA execution provider | `.onnx` | NVIDIA GPU |
| `ORT_CPU` | ONNX Runtime | `.onnx` | CPU |
| `NCNN` | ncnn (submodule) | `.param` + `.bin` | CPU (or Vulkan) |

### CUDA backend (ONNX Runtime + nvcc)

ncnn has no CUDA backend, so the GPU path uses ONNX Runtime's CUDA execution provider.
The image is copied to the GPU once; CUDA kernels compiled with nvcc letterbox the regions
(bilinear resize and gray padding; slices of the input size are copied 1:1) and write the
network input directly (BGR→RGB, 0–1, planar), a batch of regions at a time,
and after inference pick the best class per anchor and keep only the candidates above
the threshold, so just those few boxes come back to the CPU (instead of ~2.7 MB of raw
output per region). The network input and output stay in GPU memory (ONNX Runtime
IoBinding).

1. **CUDA toolkit** (nvcc) and **cuDNN 9** for your CUDA major version. In WSL2 install
   only the toolkit (`cuda-toolkit` from NVIDIA's `wsl-ubuntu` repo) and cuDNN, never a
   Linux GPU driver. With NVIDIA's apt repo: `sudo apt install libcudnn9-cuda-13`
   (or `-cuda-12`).
2. **ONNX Runtime GPU package**: download the Linux x64 GPU archive
   (`onnxruntime-linux-x64-gpu-<version>.tgz`) from
   [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and unpack
   it. ONNX Runtime 1.27 and later is built for CUDA 13; use an older release or its
   CUDA 12 variant with a CUDA 12 toolkit.
3. **Model**: export to ONNX with a dynamic batch size, so several slices run per call:

   ```bash
   libreyolo export --model LibreYOLO9t.pt --format onnx --imgsz 640 --dynamic
   ```

   The class names, input size and box format are read from the ONNX file's metadata.
4. **Build**:

   ```bash
   cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-gpu-<version>
   cmake --build build -j 2
   ```

   nvcc compiles the kernels for the GPU in the machine (CMake ≥ 3.24); otherwise pass
   e.g. `-DCMAKE_CUDA_ARCHITECTURES=89`. nvcc is found on PATH, in `$CUDA_HOME/bin`, or in
   `/usr/local/cuda*/bin` (newest version first); otherwise pass
   `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`. The program finds
   `libonnxruntime*.so` in the package directory by itself (RPATH).

```bash
./build/yolov9_sahi models/LibreYOLO9t.onnx images/ -o results/ --csv detections.csv
```

The startup line shows the GPU (`CUDA (NVIDIA ...)`) and the batch size. `-b/--batch`
sets the slices per run (default 8); `--device` picks the GPU. The first run of a new
batch size is slower while cuDNN picks its algorithms.

### ncnn backend (CPU)

```bash
cmake -B build -DSAHI_BACKEND=NCNN
cmake --build build -j 4
```

Options:

```bash
cmake -B build -DSAHI_BACKEND=NCNN -DSAHI_VULKAN=ON        # ncnn with Vulkan GPU support (run with --gpu)
cmake -B build -DSAHI_BACKEND=NCNN -DSAHI_SYSTEM_NCNN=ON -Dncnn_DIR=<prefix>/lib/cmake/ncnn   # installed ncnn
```

ncnn runs fast on CPU. **WSL2:** Vulkan in WSL2 goes through Mesa's Dozen driver (Vulkan
on Direct3D 12), which is very slow, so use the CUDA backend for the GPU there.

## Run

Arguments are, in order, the model, then one or more images and/or directories:

```
yolov9_sahi <model.onnx> <image|dir> [<image|dir> ...] [options]                  # ORT_CUDA / ORT_CPU build
yolov9_sahi <model.param> <model.bin> <image|dir> [<image|dir> ...] [options]     # NCNN build
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
| `-s, --size` | model metadata, else 640 | model input size |
| `--slice` | input size | slice size (slices are letterboxed to the input size) |
| `--overlap` | 0.20 | slice overlap ratio |
| `--full` | off | also detect on the whole image (catches objects larger than a slice) |
| `-b, --batch` | 8 | ONNX Runtime: slices per inference run (models exported with a dynamic batch) |
| `--device` | 0 | ORT_CUDA: GPU index |
| `-w, --workers` | physical cores | ncnn: slices run in parallel (OpenMP) |
| `-t, --threads` | cores / workers | ncnn: threads per slice |
| `-j, --cores` | physical cores | ncnn / ORT_CPU: total thread budget |
| `--gpu` | off | ncnn: Vulkan (needs `-DSAHI_VULKAN=ON`) |
| `--conf` | 0.25 | confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold within a slice |
| `--merge` | `nmm` | merging across slices: `nmm` merges overlapping boxes into one, `nms` keeps the best |
| `--match` | `ios` | overlap metric for the merge: `ios` (intersection over smaller box) or `iou` |
| `--match-thresh` | 0.5 | overlap above which boxes are merged |
| `--box` | model metadata, else detected | box format in the output: `xyxy` or `cxcywh` |
| `--pad` | model metadata, else `center` | letterbox placement: `center` or `topleft` |
| `--in`, `--out` | model's first input/output | input/output names |

### How SAHI works here

1. **Slices.** The image is cut into `slice × slice` slices (default: the model input
   size) overlapping by `--overlap` (default 20%); the last slice in each row/column is
   shifted back to end at the image border, so slices keep their full size. With
   `--full`, the whole image is added as one more region.
2. **Per region.** Each region is letterboxed to the input size (aspect ratio kept, gray
   padding centred, or top-left for LibreYOLO-trained checkpoints, from the model's
   metadata), normalized to 0–1 RGB and run through YOLOv9; boxes are decoded (best class
   per anchor), mapped back to the image and de-duplicated within the region with
   class-aware NMS.
3. **Merge.** An object cut by a slice edge appears as a full box from one slice plus a
   fragment from its neighbour, or as two halves. As in the SAHI library, the default
   merge is greedy NMM with the IOS metric: boxes of the same class whose intersection
   covers more than half of the smaller box are merged into their union, keeping the
   best score. Use `--merge nms --match iou` for plain NMS.

Output is deterministic regardless of the number of threads or the batch size. SAHI works
best when objects are smaller than a slice; add `--full` when the images also contain
objects much larger than a slice.

### Threading

With ncnn, the program runs **one slice per physical core, with ncnn single-threaded**.
SAHI produces many equal slices, and running them side by side scales almost linearly,
while ncnn's own per-layer threading scales worse on a single 640×640 slice and nesting the
two levels costs extra. Measured on 2 cores with YOLOv9-C, 35 regions:

| Setup | Inference | vs 1 core |
|---|---|---|
| 1 core | 32.0 s | 1.0× |
| 1 slice × 2 ncnn threads | 21.8 s | 1.47× |
| 2 slices × 1 thread (default) | 16.1 s | 1.99× |

Details:

- The split is fixed at startup: ncnn's convolution layers fix their thread count when
  the model is loaded. `-w` (slices in parallel) and `-t` (ncnn threads per slice) override
  it; `-j` sets the total budget (default: physical cores; hyperthreads add little here).
- Images that fit in a single slice can't use slice parallelism. For those, a second copy
  of the model with all cores is loaded the first time such an image appears.
- All slices share one copy of the model weights; each worker has its own ncnn memory pools
  (by default ncnn's extractors share one locked pool, which serializes concurrent slices).
- With several slices in flight, ncnn threads don't spin-wait between layers, so idle
  threads don't take cores from other slices.
- The next image is decoded and the previous result is drawn and saved in the background
  while the current image runs.

# yolov9-sahi

YOLOv9 inference on large images with the **SAHI** method (Slicing Aided Hyper Inference), running on
[MNN](https://github.com/alibaba/MNN) with a **CUDA** or **Vulkan** backend (one per build), or on the CPU.

This is the `mnn` branch. Other branches: `main` (tiling, ncnn), `cuda-backend` (tiling, ONNX Runtime CUDA),
`sahi-cuda` (SAHI, ONNX Runtime CUDA).

You only need the model's **.onnx** file. On first use it is converted to MNN's format (`model.mnn`, next to the
`.onnx`) by MNN's own converter, which is built together with the program.

## How it works

1. **Slicing** – the image is cut into overlapping slices of the model's input size (default 640×640, 20 % overlap;
   the last slice in each row/column is moved back so it ends at the image border). With `--full`, the whole image is
   added as one more region so large objects are found too.
2. **Preprocessing** – each region is letterboxed as YOLOv9 expects: bilinear resize (as OpenCV `INTER_LINEAR`),
   gray (114) padding, centred (or top-left for LibreYOLO-trained checkpoints, from the model's metadata), RGB
   scaled to 0–1.
3. **Inference** – regions are run in batches (`--batch`, default 8) on the selected MNN backend.
4. **Per-region NMS** – class-aware NMS inside each region (`--iou`).
5. **Merging across slices** – as in the SAHI library: greedy non-maximum *merging* (NMM) with the IOS metric
   (intersection over the smaller box), so an object cut by a slice edge becomes one box again. `--merge nms` and
   `--match iou` give plain NMS instead.

Preprocessing, decoding and un-letterboxing are parallelised with OpenMP; the next image is decoded and the previous
result is drawn and saved in the background.

## Model

The model comes from [LibreYOLO](https://github.com/LibreYOLO/libreyolo) (MIT license), exported to ONNX with a
dynamic batch size, so several slices run per call:

```bash
pip install libreyolo
libreyolo export --model LibreYOLO9t.pt --format onnx --imgsz 640 --dynamic   # or your own trained checkpoint
```

The class names, input size, box format and letterbox placement are read from the ONNX file's metadata (also when
you pass the converted `.mnn` and the `.onnx` of the same name is next to it). LibreYOLO's YOLOv9 outputs one
`(4 + classes) × anchors` tensor: `x1, y1, x2, y2` in input pixels, then sigmoid class scores.

Without LibreYOLO metadata, the box format (`x1,y1,x2,y2` or centre/width/height) is detected from the model's
output, the image is centred, and class names come from `--names`. `--box`, `--pad`, `--in` and `--out` override
the detection. A model exported with a fixed batch of 1 still works, one slice per run.

## Requirements

- CMake ≥ 3.18, a C++17 compiler, OpenCV (core, imgproc, imgcodecs, dnn), OpenMP
- **CUDA backend:** the CUDA toolkit (`nvcc`). cuDNN is **not** needed.
- **Vulkan backend:** nothing at build time (MNN ships its own Vulkan headers); a Vulkan driver at run time.

```bash
sudo apt install build-essential cmake libopencv-dev
```

## Build

```bash
git clone --recursive -b mnn https://github.com/detu/yolov9-sahi.git
cd yolov9-sahi
# (already cloned without --recursive, or switched branches?  git submodule update --init)

# CUDA (+ CPU), the default
cmake -B build
cmake --build build -j 2

# or Vulkan (+ CPU)
cmake -B build-vulkan -DYOLOV9_VULKAN=ON
cmake --build build-vulkan -j 2

# or CPU only
cmake -B build-cpu -DYOLOV9_CUDA=OFF
```

Only **one** GPU backend can be built: `YOLOV9_CUDA` and `YOLOV9_VULKAN` cannot both be `ON` (CMake stops with an
error; turning Vulkan on switches the CUDA default off). The CPU backend is always included, so `--backend cpu`
works in every build. Use a separate build directory per backend. Builds are Release by default.

The MNN submodule (pinned at 3.6.2) is configured and built automatically, including `MNNConvert`. The first build
takes a while; on machines with little RAM keep `-j` low (`-j 2`) — an out-of-memory kill shows up as
`Error 137` / `Killed`.

| CMake option | Default | Meaning |
|---|---|---|
| `YOLOV9_CUDA` | `ON` | Build MNN's CUDA backend |
| `YOLOV9_VULKAN` | `OFF` | Build MNN's Vulkan backend |
| `YOLOV9_CUDA_ARCH` | *(nvidia-smi)* | GPU compute capability to compile for, e.g. `8.9` (RTX 40xx / RTX Ada laptop GPUs), `8.6` (RTX 30xx), `12.0` (RTX 50xx) |
| `YOLOV9_CUDA_NATIVE_ARCH` | `ON` | Compile the CUDA kernels only for that one GPU (much faster build) |
| `YOLOV9_NVCC` | *(search)* | Path to `nvcc` |

`nvcc` is searched on `PATH`, in `$CUDA_HOME`, `$CUDA_PATH`, `/usr/local/cuda` and `/usr/local/cuda-*` (newest
first), so it does not need to be on `PATH` (common on WSL2).

## Usage

```bash
./build/yolov9_sahi <model.onnx|model.mnn> <image|dir> [<image|dir> ...] [options]
```

```bash
# One image -> output.jpg
./build/yolov9_sahi models/LibreYOLO9t.onnx big_image.jpg

# A folder (recursively) -> out/<name>_det.jpg, plus all detections as CSV
./build/yolov9_sahi models/LibreYOLO9t.onnx images/ -r -o out --csv detections.csv

# Vulkan build, whole image as an extra region, own class names
./build-vulkan/yolov9_sahi models/LibreYOLO9t.onnx images/ --full -n classes.txt
```

The model is loaded once and every image is processed with it. Output boxes are drawn with a label (class name and
score) on a green background.

| Option | Default | Meaning |
|---|---|---|
| `-o, --output` | `output.jpg` / `output/` | Output file (one image) or directory (several images) |
| `--csv <file>` | | Write all detections (`image,class_id,class_name,score,x1,y1,x2,y2`) |
| `-n, --names <file>` | model metadata | Class names, one per line: line 1 = class 0, line 2 = class 1, … |
| `-r, --recursive` | | Scan directories recursively |
| `-d, --backend` | `auto` | `cuda` or `vulkan` (whichever was built), or `cpu`; `auto` = the GPU backend if it works here, else CPU |
| `--precision` | `normal` | `normal`, `high`, or `low` (fp16 where the backend supports it) |
| `-b, --batch` | `8` | Regions per inference run (models exported with a fixed batch fall back to 1) |
| `-t, --threads` | all cores | CPU backend threads and pre/post-processing threads |
| `--fp16` | | Convert the `.onnx` with fp16 weights (`model.fp16.mnn`) |
| `--reconvert` | | Convert again even if the `.mnn` is up to date |
| `-s, --size` | model | Model input size |
| `--slice` | input size | Slice size |
| `--overlap` | `0.2` | Slice overlap ratio |
| `--full` | | Also detect on the whole image |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold inside a region |
| `--merge` | `nmm` | Cross-slice merging: `nmm` (merge boxes) or `nms` (keep the best) |
| `--match` | `ios` | Overlap metric for merging: `ios` or `iou` |
| `--match-thresh` | `0.5` | Overlap above which boxes are merged |
| `--preprocess` | metadata / `yolo` | `yolo` (RGB, 0–1, OpenCV-style bilinear resize) or `rtmdet` (BGR, mmdet mean/std) |
| `--pad` | metadata / `center` | Letterbox placement: `center` or `topleft` (LibreYOLO-trained checkpoints) |
| `--box` | metadata / detected | Box format of the output: `xyxy` or `cxcywh` |
| `--in`, `--out` | | Input / output tensor names |

## Troubleshooting

- **`nvcc not found`** – install the CUDA toolkit or pass `-DYOLOV9_NVCC=/usr/local/cuda/bin/nvcc`; or build the
  Vulkan backend instead: `-DYOLOV9_CUDA=OFF -DYOLOV9_VULKAN=ON`.
- **`Error generating file ... MNN_Cuda_Main_generated_*.cu.o`** – nvcc failed on one of MNN's CUDA files; the real
  message is a few lines above it (rebuild with `cmake --build build -j 1 2>&1 | tee build.log` to see it clearly).
  - `Unsupported gpu architecture 'compute_60'` (CUDA 13): the GPU architecture was not known while configuring.
    It is now read from `nvidia-smi`; otherwise pass it, e.g. `-DYOLOV9_CUDA_ARCH=8.9`, in a **new** build directory.
  - `cutlass/matrix.h: ... has no member named 'set_slice3x3'` (GCC 14 and newer): a typo in MNN's bundled CUTLASS
    that newer GCC rejects. Handled automatically (`-Wno-template-body` is passed to the host compiler).
  - `Killed` / `Error 137` / `cicc died`: out of memory; MNN's CUTLASS kernels are heavy to compile. Use `-j 1`.
- **`Warning: cuda backend not available here; MNN uses CPU`** – the CUDA driver/GPU is not visible (check
  `nvidia-smi`). Under WSL2, `/usr/lib/wsl/lib` must be on the library path.
- **Vulkan under WSL2** uses the Dozen (D3D12) driver, which works but is slower than CUDA.
- **Slower after a while on a laptop** – usually thermal/power throttling of the GPU; check with
  `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv -l 1`.

## License

MNN (the `MNN/` submodule) is licensed under Apache-2.0. YOLOv9 models from LibreYOLO are MIT-licensed.

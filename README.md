# Quality-Aware Deferred Recognition and Two-Tier Voting for Video Licence Plate Recognition

Code and trained detector for the paper *"Quality-Aware Deferred Recognition and Two-Tier Voting for Video Licence Plate Recognition under Adverse Conditions"*.

The pipeline reads a traffic video and returns one licence plate string per tracked plate:

1. **Detection** – a single-class (`LP`) YOLOv8n detector localises plates in each frame.
2. **Tracking** – ByteTrack assigns a persistent identity to each plate.
3. **Quality-aware buffering** – every crop is scored on sharpness, exposure and size; the five best crops per plate are kept.
4. **Deferred recognition** – TrOCR (`microsoft/trocr-base-printed`) is run only once per plate, after the plate has left the scene or has been tracked long enough.
5. **Two-tier voting** – exact-match voting over the readings, with character-level positional voting as the fallback.

## Repository contents

| File | Description |
|---|---|
| `video_analyzer_voting.ipynb` | The pipeline described in the paper (tracking, quality-aware buffer, deferred TrOCR, two-tier voting). |
| `VideoAnalyzer.ipynb` | Earlier per-frame version without tracking, buffering or voting. Kept for reference only. |
| `best_license_plate_model.pt` | Detector weights loaded by the notebooks. |
| `lp_train_20260313_213934-…zip` | Detector training run: `args.yaml`, `results.csv`, training curves, confusion matrix and `weights/`. |

## How to run

The notebooks are written for Google Colab with a GPU runtime.

1. Place the files in Google Drive in this layout:

   ```
   DATA/
   ├── Models/
   │   └── best_license_plate_model.pt
   ├── Input_Videos/
   │   └── your_video.mp4
   └── Output_Videos/
   ```

2. Open `video_analyzer_voting.ipynb` in Colab and run the cells in order.
3. Set `VIDEO_SOURCE`, `OUTPUT_VIDEO_PATH` and `MODEL_PATH` in the second code cell to your own paths.
4. The final cell writes the annotated video and prints one reading per tracked plate together with the voting method used.

## Pipeline parameters

| Parameter | Value |
|---|---|
| Frame size | 1280 × 720 |
| Detection confidence threshold | 0.5 |
| Tracker | ByteTrack (`bytetrack.yaml`) |
| Minimum sharpness (variance of Laplacian) | 100 |
| Maximum fraction of clipped pixels | 0.15 |
| Minimum crop size | 100 × 40 px |
| Quality score | 0.5 · sharpness + 0.3 · exposure + 0.2 · size |
| Buffer size | 5 crops per plate |
| Recognition trigger | plate unseen for 15 frames, or tracked for 90 frames |
| TrOCR precision | FP16 on GPU |

## Requirements

```
ultralytics==8.3.152
transformers==4.52.4
opencv-python==4.11.0.86
torch
```

## Dataset

The video dataset is not included in this repository. It is available from the corresponding author on reasonable request.

## Licence

The detector is built with Ultralytics YOLO, which is released under AGPL-3.0; this repository is released under the same licence.

## Citation

If you use this code, please cite the paper (details will be added on publication).

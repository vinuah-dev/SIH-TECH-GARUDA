# Where we stopped — 29 Aug 2026 (updated)

## DONE since: the rescue pass is implemented and on by default

`PlateDetector.rescue_crops` (app/anpr/detector.py). The whole-frame pass runs
as before; when it finds no plate on a vehicle, that vehicle alone is cropped,
enlarged 3x and searched again. Measured on the Delhi 1080p clip:

| | plates found | in the full pipeline |
| --- | --- | --- |
| rescue off | 36/171 (21.1%) | crops on **2 of 71** vehicles |
| rescue on  | 75/171 (43.9%) | crops on **15 of 66** vehicles |

Roughly double the recall in isolation, eight times the plate evidence in the
pipeline (the pipeline gains more because ANPR retries a track across frames).

**The fps figures I took are not trustworthy and must be redone.** Two 400-frame
runs gave 3.3 fps with rescue ON and 2.4 fps with it OFF - i.e. the expensive
path measured *faster*, which cannot be right. Something else moved between the
runs (model warm-up, thermal state, the OS). Redo this properly - several runs,
alternating order - before quoting any number. In isolation the rescue pass
clearly did cost time: 41s -> 85s over the same frames.

TODO next on this: tests for `_rescue` (coordinate mapping back to frame
coordinates is the part that would silently file a crop from the wrong part of
the picture), a `--no-plate-rescue` flag, and a README correction - it still
says "Localisation: solved", which these measurements contradict.

## The finding that led to it

The plate model runs on the **whole frame** (one pass per frame, for speed) and
throws away any plate under `min_width = 60` px. On the user's real phone
footage (848x478) that is almost everything.

Measured on video C (daylight, head-on), 308 vehicles wide enough to try:

| approach | plates found |
| --- | --- |
| whole frame, `min_width=60` (**what ships today**) | 15 (4.9%) |
| vehicle crop upscaled x3, `min_width=12` | **135 (43.8%)** |

**Nine times better.** Median plate is 41 px in the original frame — well under
the 60 px floor, which is why they were being discarded rather than missed.

Not yet done: this costs one model pass per vehicle instead of one per frame,
so it must be measured for fps before it becomes the default. Likely shape of
the fix: keep the whole-frame pass, and fall back to the cropped pass only for
vehicles where the frame pass found nothing.

## Real-footage numbers (were never measured before these videos)

Video A, dusk, handheld, brightness 83/255, sharpness 371:
- 1471 frames, 8420 detections, 4.2 fps
- 151 vehicles logged, 75 duplicates suppressed — the pass log works
- plate crop on 4 of 151 (2.6%), **0 plates read**

Video C, daylight, brightness 145/255, sharpness 805: plates found 4.9%.
So darkness is not the main cause — resolution and the `min_width` floor are.

Lowering the plate detector's confidence does NOT help (0.30 -> 0.04 moved
1.3% -> 5.3%, and 0.04 is mostly noise).

**The README currently says "Localisation: solved". On this footage it is not.
That claim needs correcting once the fix above is measured.**

## Recogniser comparison (on 400 labelled Indian plates, clean web photos)

| recogniser | exact | characters |
| --- | --- | --- |
| EasyOCR (**what ships today**) | 34.5% | 71.2% |
| CRNN trained here, epoch 50 | 46.1% | 75.1% |
| **PaddleOCR** | **60.2%** | **80.3%** |

PaddleOCR is the clear winner and is installed (needs `enable_mkldnn=False` on
this machine — oneDNN kernels abort otherwise). **It is NOT wired into the app
yet**, only into `tools/plate_benchmark.py --engine paddle`. Wiring it in is the
single biggest accuracy win available and needs no new data.

The CRNN run died at epoch 50 of 150 with no error. `models/plate-reader.pt`
holds that epoch and scores 46.1%, verified by reloading it.

## State of the tree

- 781 tests pass
- `serve.py` was silently duplicated twice; fixed, and `tests/test_sources_intact.py`
  now guards every source file against it (verified it catches a real duplicate)
- Map/geolocation backend done: `app/geo/`, `/api/map`, `/api/map/cameras/{id}`
- Pass log done: every vehicle logged, 10 s de-duplication

## Datasets on disk (gitignored, ~400 MB)

- `data/datasets/indian-plates` — 1629 plates with real text labels
- `data/datasets/crops` — 1380 train / 311 val crops, split by plate number
- `data/datasets/generic-plates` — detection only, label is just "licence"
- `data/datasets/chars` — 28x28 typed glyphs; judged not useful for plates

# IBVAP — Intelligent Border Video Analytics Platform

**SIH 2026 · Problem Statement 26187**
Ministry of Home Affairs · Sashastra Seema Bal (SSB), Police II Division

A software intelligence layer that runs on top of **existing IP CCTV infrastructure**,
so border posts get analytics without replacing cameras with expensive smart hardware.

```
VIDEO → DETECTION → TRACKING → ZONES → CONTEXT → BEHAVIOUR → RISK → ALERT → EVIDENCE
```

**926 automated tests.** Every claim below is qualified by whether it has been run
against real footage or only against a test rig — see
[What is proven, and what is not](#what-is-proven-and-what-is-not).

---

## What it does

* **Detects** people and vehicles (car, truck, bus, motorcycle, bicycle) on a still,
  a video file, a webcam or a live RTSP/HTTP camera.
* **Tracks** them with stable IDs, and **recovers a track after an occlusion** by
  appearance, so a person who steps behind a pillar is still the same person.
* **Fences** areas with polygon zones that know what they are fencing — a vehicle lane
  is a fence for pedestrians but not for the trucks it exists to admit.
* **Measures** speed, heading, dwell time, distance to the fence and proximity to the
  camera; in metres per second on a calibrated camera.
* **Names behaviours**: loitering, border-facing movement, erratic movement, night
  movement, rapid movement, and camera tampering.
* **Watches itself**: a camera that stops answering is treated as a possible tampering
  event, not just an outage.
* **Recognises the people who belong** — an enrolled guard on patrol stops reading like
  an intruder, so the alerts that remain mean something.
* **Scores risk** 0–100, where every point is attributable to a named factor.
* **Reads number plates** off vehicles, pools reads across cameras, and checks them
  against a vehicle registration extract — flagging a plate that does not match the
  vehicle carrying it.
* **Keeps evidence**: an annotated still, a video clip covering the seconds either side
  of the alert, and the plate crop.
* **Follows a subject between cameras** — vehicles by plate, people by appearance,
  with the weaker of the two labelled as inferred.
* **Runs many cameras in one process**, sharing one loaded model.
* **Serves** a live dashboard and a JSON API.

---

## Quick start

```bash
pip install -r requirements.txt
```

The fastest way to see the whole pipeline — no model download, no camera, works offline:

```bash
python main.py --source synthetic --detector sim --force-night
```

A threat that escalates as it unfolds:

```bash
python main.py --source synthetic --detector sim --synthetic-frames 360 --linger --force-night
```

The cloned-plate story — a bus wearing a motorcycle's plate, crossing a fence:

```bash
python tools/make_vehicle_demo.py
```

```bash
python main.py --source data/samples/vehicle_demo.mp4 --detect all --registry config/vehicle-registry.json --force-night --db
```

The plate on that clip is drawn on, not photographed. It demonstrates the pipeline end
to end; it does not demonstrate that OCR reads real plates — see
[ANPR: an honest account](#anpr-an-honest-account).

Live camera, with an annotated window:

```bash
python main.py --source webcam:0 --view --force-night
```

With more than one camera, `--view` draws a **single window** holding the whole fleet —
a 2x2 wall for four cameras, not four windows to find and keep on top. Tile order is
fixed by camera id, so a camera that connects late takes its own place instead of
reshuffling the others under the operator's eye.

Every camera in the fleet, one process, plus the dashboard on http://127.0.0.1:8000:

```bash
python serve.py --view
```

The API has **no authentication**. It binds to loopback by default for that reason;
`--host 0.0.0.0` warns before exposing it. Put it behind a VPN or an authenticating
reverse proxy before it leaves the machine.

---

## Sample alert

```
==================================================================
                          SECURITY ALERT
==================================================================

Type         : VIRTUAL FENCE INTRUSION
Object       : BUS
Camera       : CAM-01
Track ID     : P001
Confidence   : 93%
Zone         : RESTRICTED (RESTRICTED)
Plate        : DL 8 CAF 5030
Registered   : Motorcycle - Black Hero Splendor  [MISMATCH]
Behaviours   : BORDER-FACING MOVEMENT, NIGHT MOVEMENT, PLATE MISMATCH
Severity     : CRITICAL
Risk Score   : 100/100
Time         : 2026-08-25 23:31:39

Reason:
Restricted-zone entry detected. Multiple contextual signals indicate
suspicious activity.

Contributing factors:
  +72  RESTRICTED ZONE - No-entry strip along the fence line (bus, weight x1.2)
  +10  DETECTION CONFIDENCE - model confidence 93%
  +6   NIGHT-TIME - movement at 23:31 (night window)
  +8   BORDER-FACING MOVEMENT - closing on RESTRICTED heading E for 38 frames
  +4   NIGHT MOVEMENT - moving at 0.07 w/s during the night window
  +35  PLATE MISMATCH - plate is registered to a motorcycle, camera sees a bus

Evidence     : data/evidence/20260825-233139_CAM-01_4B9C4F4733EF.jpg
Clip         : data/evidence/20260825-233139_CAM-01_4B9C4F4733EF.mp4
Plate img    : data/evidence/20260825-233139_CAM-01_4B9C4F4733EF-plate.jpg
```

---

## What is proven, and what is not

This matters more than the feature list. Nothing below is a projection.

### Verified on real footage or real hardware

| Capability | How it was verified |
| --- | --- |
| Person detection | Live phone camera, a room of people, 8.4 fps |
| Live IP camera ingestion | Phone running IP Webcam over WiFi |
| Vehicle detection | Delhi traffic clip — 186 vehicle detections across 40 sampled frames |
| Plate localisation | Same clip — 30 plate detections across 40 frames, median 129 px |
| Multi-camera in one process | Three cameras, one shared model, one event store |
| Dead-camera handling | Unroutable RTSP host: fails in 10.5 s with a clear reason, not a hang |
| Event store | Three concurrent processes and four threads writing, zero loss |

### Verified against a test rig only

Re-identification through occlusion, camera-tampering proximity, cross-camera plate
handoff, ground-plane calibration, and evidence clips are covered by tests using
synthetic or composited inputs. They work; they have not met a real border.

### Does **not** work yet

**Automatic number plate reading.** On the Delhi traffic clip, zero plates were read
with confidence. The full diagnosis is in
[ANPR: an honest account](#anpr-an-honest-account) — it is worth reading before
demonstrating this feature.

What ANPR *does* do reliably: locate the plate with a purpose-trained model, reject
motion-blurred frames, save the plate crop as evidence, and refuse to write a number it
is not sure of.

---

## Project structure

```
ibvap/
├── app/
│   ├── pipeline.py            # wires every stage together, owns the run loop
│   ├── runner.py              # many cameras, one process, one shared model
│   ├── config.py              # SurveillanceConfig - one session's settings
│   ├── calibration.py         # ground-plane homography: image -> metres
│   ├── ui.py                  # branding, colour, thread-safe console output
│   │
│   ├── video/
│   │   ├── sources.py         # image / video / webcam / synthetic ingestion
│   │   ├── stream.py          # RTSP & HTTP with reconnect and link health
│   │   └── annotate.py        # zone, box and HUD drawing
│   │
│   ├── detection/
│   │   ├── base.py            # Detection model, Detector protocol
│   │   ├── classes.py         # what is detectable, and how it is grouped
│   │   ├── yolo.py            # YOLOv8 backend
│   │   ├── tracker.py         # IoU tracking with appearance re-identification
│   │   ├── appearance.py      # colour descriptors for re-ID and handoff
│   │   └── simulator.py       # scripted actor for model-free demos and CI
│   │
│   ├── zones/                 # polygon fences, geometry, per-object rules
│   ├── context/               # speed, heading, dwell, approach, apparent size
│   ├── behaviour/
│   │   ├── engine.py          # loitering, border-facing, erratic, tampering
│   │   └── camera.py          # link loss treated as a security event
│   ├── risk/engine.py         # explainable 0-100 scoring
│   │
│   ├── anpr/
│   │   ├── engine.py          # plate reading, throttling, multi-frame voting
│   │   ├── plate.py           # where on a vehicle to look
│   │   ├── text.py            # plate format validation and repair
│   │   └── ledger.py          # plate reads pooled across the fleet
│   ├── registry/vehicles.py   # registration lookup and plate/vehicle mismatch
│   │
│   ├── alerts/                # events, console rendering, JSONL log, clips
│   ├── store/database.py      # queryable SQLite event store
│   └── server/                # FastAPI dashboard, REST and WebSocket
│
├── config/                    # zones, cameras, vehicle registry
├── data/                      # samples, evidence, event logs
├── tests/                     # 926 tests, no model or camera required
├── main.py                    # CLI entrypoint
└── serve.py                   # server entrypoint
```

The three analysis stages are deliberately separate, because that separation is what
makes an alert explainable:

| Stage | Job | Example |
| --- | --- | --- |
| Context | **Measures** | `dwell 14s`, `heading E`, `1.65 m/s` |
| Behaviour | **Names** | `LOITERING`, `BORDER-FACING MOVEMENT` |
| Risk | **Scores** | `+14 LOITERING` → `90/100 CRITICAL` |

---

## Running it

### Sources

```bash
python main.py --source synthetic --detector sim     # no model, no camera
python main.py --source data/samples/clip.mp4        # a video file
python main.py --source photo.jpg                    # a still
python main.py --source webcam:0                     # local camera
python main.py --source rtsp://user:pass@10.0.0.5:554/stream1
python main.py --source http://192.168.1.5:8080/video   # IP Webcam on a phone
```

### What to watch for

```bash
python main.py --source clip.mp4 --detect person          # default
python main.py --source clip.mp4 --detect vehicle
python main.py --source clip.mp4 --detect all
python main.py --source clip.mp4 --detect person truck
```

### Fleet

`config/cameras.json` lists the cameras. Each may have its own zones and override any
setting. A camera that is not answering is skipped at startup with a clear reason rather
than stalling the fleet.

```bash
python main.py --cameras config/cameras.json --view --db
```

### Server

```bash
python serve.py                 # dashboard + API
python serve.py --view          # also opens a window per camera
python serve.py --host 0.0.0.0 --port 8080
```

| Endpoint | Returns |
| --- | --- |
| `/` | Operator dashboard, live over WebSocket |
| `/api/status` | Per-camera frames, alerts, fps, link state |
| `/api/summary` | Totals by severity and camera |
| `/api/events` | Recent events, filterable by camera and severity |
| `/api/events/{id}` | One event in full |
| `/api/behaviours` | How often each behaviour has fired |
| `/api/incidents` | Events grouped into incidents, not one row per alert |
| `/ws` | Live alert push |

### Useful flags

`--view` `--db` `--registry PATH` `--stride N` `--limit N` `--loop` `--confidence`
`--confirm-frames` `--cooldown` `--force-night` `--no-reid` `--no-anpr` `--no-clips`
`--reconnect-attempts` `--verbose` `--no-color`. Full list: `python main.py --help`.

---

## Zones

Polygons in **normalized 0–1 coordinates**, so one fence file works at any resolution.
Membership is tested at the person's **ground point** — the bottom-centre of the box,
where their feet are — because a standing person's centroid crosses a ground-plane fence
well before they do.

```json
{
  "zones": [
    {
      "name": "VEHICLE-LANE",
      "kind": "RESTRICTED",
      "base_risk": 60,
      "objects": ["PERSON"],
      "description": "Vehicle lane - no pedestrians",
      "polygon": [[0.35, 0.45], [0.75, 0.45], [0.90, 1.00], [0.20, 1.00]]
    }
  ]
}
```

`kind` is `RESTRICTED`, `WATCH` or `PATROL`; only kinds named in `--alert-on` raise
alerts. `objects` decides which object categories the fence applies to. Overlapping
zones resolve to the highest priority. Concave polygons work.

### Ground-plane calibration (optional)

Four surveyed reference points give the camera a homography, after which speed reads in
m/s and distances in metres. Without it everything is in frame widths, which means the
same threshold means different things on different cameras.

```json
"calibration": {
  "unit": "m",
  "points": [
    { "image": [0.10, 0.56], "world": [0.0, 40.0] },
    { "image": [0.95, 0.56], "world": [12.0, 40.0] },
    { "image": [1.00, 0.99], "world": [12.0, 6.0] },
    { "image": [0.05, 0.99], "world": [0.0, 6.0] }
  ]
}
```

`config/zones-calibrated.json` is a worked example. **Its numbers are illustrative** —
chosen to give the bundled demo plausible metric behaviour, not surveyed from a real
camera.

---

## How the risk score is built

| Factor | Points | Source |
| --- | --- | --- |
| Zone base risk | `RESTRICTED` +60, `WATCH` +30, scaled by object weight | Zone config |
| Object weight | bicycle ×0.85, person ×1.0, car ×1.10, truck ×1.20 | Class registry |
| Detection confidence | −10 … +10 around a 60 % pivot | Detector |
| Night-time presence | +6 inside the night window | Scene context |
| `NIGHT MOVEMENT` | +4 moving during that window | Behaviour |
| `BORDER-FACING MOVEMENT` | +8 after 5 consecutive frames closing on a fence | Behaviour |
| `LOITERING` | +1.5/s past a 5 s grace, capped +20 | Behaviour |
| `ERRATIC MOVEMENT` | +6 when heading variance exceeds 0.45 | Behaviour |
| `RAPID MOVEMENT` | +6 above 2.0 m/s (11 m/s for vehicles) — calibrated only | Behaviour |
| `CAMERA TAMPERING` | +30 when a person fills 65 % of frame height | Behaviour |
| `PLATE MISMATCH` | +35 plate registered to a different vehicle class | Registry |
| `WATCHLISTED VEHICLE` | +45 registry status stolen or blacklisted | Registry |
| `UNREGISTERED PLATE` | +5 not in the registry extract | Registry |
| `CAMERA LINK LOST` / `OFFLINE` | +40 / +55 | Camera integrity |

Severity: `CRITICAL` ≥ 90, `HIGH` ≥ 65, `MEDIUM` ≥ 40, else `LOW`. The score is the sum
of the printed factors clamped to 0–100 — enforced by a test, because an operator has to
be able to see *why* an alert fired.

Three calibration choices are deliberate:

* **Night is split** into presence (+6) and movement (+4). Someone standing in a
  restricted zone at night still carries the night signal; someone moving carries more.
* **Loitering outweighs border-facing.** Walking in at night scores 88 (`HIGH`).
  Stopping ends the border-facing signal, but loitering then climbs past 90 (`CRITICAL`).
  Someone who enters and **stays** must end up above someone who walks through.
* **Absence from the registry barely counts** (+5). An extract is always incomplete;
  crying wolf at every unlisted vehicle would be useless.

---

## Alerting policy

Three rules keep the console usable without hiding a worsening situation:

* **Confirmation frames** — a detection must sit inside the zone for 3 consecutive
  frames, which suppresses single-frame detector flicker on a boundary.
* **Cooldown** — the same track cannot re-alert on the same zone for 30 footage seconds.
* **Escalation overrides the cooldown** — an alert whose severity is *higher* than the
  one already raised always gets through. An intruder who stops and loiters must reach
  the operator.

Camera tampering is a separate trigger that is **not** tied to a zone: someone reaching
for the lens is a threat wherever they stand, and blinding the camera defeats every zone
rule anyway.

### Incidents, not alerts

Four rows saying the same person is still inside the fence is four chances to stop
reading them. `/api/incidents` groups events that concern the same track on the same
camera within a time window — or, for vehicles, that share a plate, which links sightings
across cameras. Each incident carries its worst moment, every behaviour seen during it,
and the ids of its events.

This is a **read-side view**. Nothing about what is detected or stored changes, so an
incident can always be expanded back into the events it was built from.

```
7 alerts  ->  3 incidents
   4 events  peak  95 CRITICAL  CAM-01           BORDER-FACING MOVEMENT, LOITERING
   2 events  peak  92 CRITICAL  CAM-01, CAM-02   plate MH12AB1234, PLATE MISMATCH
   1 event   peak  72 HIGH      CAM-01
```

---

## ANPR: an honest account

Number plate reading was built end to end — localisation, blur rejection, OCR, format
validation, multi-frame voting, cross-camera pooling, registry verification — and then
measured against a real Delhi traffic clip. Localisation works **once a plate is large
enough in frame**, and the section below records how much work that qualifier turned out
to hide. Reading now works on 73.2% of real Indian plates, after three wrong turns.
The route matters more than the result.

### Localisation, corrected twice — and then corrected again

An earlier version of this README said localisation was *solved*, on the strength of the
table below. Phone footage from a real street later showed that claim was worth far less
than it looked: on 848×478 video the plate model found a plate on **2.8%** of vehicles.
The table was measured on frames where plates were already large, which is not a
property of the method — it is a property of that clip.

The cause was not the model and not its confidence threshold (lowering it from 0.30 to
0.04 moved 1.3% to 5.3%, and 0.04 is mostly noise). It was that the model ran **once on
the whole frame** — cheap, and correct when plates are large — and discarded anything
under a 60 px floor. At 848×478 a plate is about 40 px. They were being thrown away, not
missed.

**The fix: when the frame pass finds no plate on a vehicle, that vehicle alone is
cropped, enlarged 3×, and searched again.**

| On the Delhi clip | plates located |
| --- | --- |
| frame pass only | 36 of 171 (21.1%) |
| plus the close-up rescue, as first written | 75 of 171 (43.9%) |
| **plus the rescue, after checking what it found** | **52 of 171 (30.4%)** |

**The middle row is a mistake worth keeping visible.** It was reported as a doubling of
recall, and it was not: looking at the crops showed the extra finds included a video
watermark stretched across a bus and a `TATA` badge. Counting detections without
inspecting them is not measuring recall, and the error was mine.

What fixed it is geometry rather than pixels. Handed a magnified, blurred vehicle crop —
nothing like the scenes it was trained on — the model returns boxes covering most of the
crop. A plate is a small part of the vehicle carrying it and much wider than it is tall,
so a box wider than half the vehicle, or outside roughly 1.6:1 to 7:1, is rejected before
anything reads it. That removed 23 of the 39 rescue finds.

The honest gain is **21.1% to 30.4%**, and the crops that survive are plates — including
two-line ones.

It costs a second model pass for each vehicle the cheap pass missed. Measured over
alternating 300-frame runs: **3.2 / 2.9 fps with it on against 4.1 / 3.1 off**, so
roughly 15% — and the spread between identical runs is wide enough that 15% is an
estimate, not a figure to quote precisely. `--no-plate-rescue` turns it off.

*(An earlier single pair of runs appeared to show the rescue pass being* faster*, which
cannot be true. That reading was noise from an unwarmed model and was discarded rather
than reported.)*

### The frame-wide pass: replacing the approach twice

| Approach | Result on the same 40 sampled frames |
| --- | --- |
| Contour and aspect-ratio search | 43 candidates, **0 were plates** |
| … plus character counting | 7 candidates, **0 were plates** |
| Purpose-trained plate model (5.5 MB) | **30 plate detections**, median 129 px wide |

A vehicle carries many plate-shaped rectangles — grille slats, bumper shadows, badge
recesses — and shape alone cannot tell them from the plate. OCR on those regions returned
`LA`, `JIA`, `WW`. Counting glyph-shaped blobs inside each candidate cut 43 to 7, and all
7 were still wrong. A model trained on plates finds them immediately.

**That fixes the evidence, which is a real operational gain.** The saved plate crop used
to be as likely to show a grille as a plate; it now shows the plate.

```bash
python tools/fetch_plate_model.py
```

The model is optional — without it the engine falls back to searching the whole vehicle
for text. It is a third-party model from Hugging Face: **check its licence before
deploying operationally.**

### Reading: measured against 400 real Indian plates

The earlier verdict here was "reading does not work", reached by watching a few dozen
frames of one video. That was wrong, and wrong in the way eyeballing usually is — it
could not tell *broken* from *nearly right*.

An annotated dataset of Indian number plates settled it. Every plate is cropped at its
annotated box and handed to the reader, so localisation is removed and recognition is
measured alone, under conditions kinder than any camera provides:

```bash
python tools/plate_benchmark.py --limit 400
```

| | Single read |
| --- | --- |
| Whole plate exactly right | **34.5 %** |
| Characters right, position by position | **71.2 %** |
| Returned nothing at all | 1.2 % |

71 % of characters is not a recogniser that cannot read this font. It is one that reads
most of it and slips on a handful of shapes. Counting every mistake across 400 plates
says which:

```
0 -> O  x22     M -> H  x11     4 -> 1  x9      A -> 4  x7
1 -> I  x7      T -> I  x7      4 -> A  x6      7 -> 2  x5
```

Those split cleanly in two, and the split decides what is fixable here.

**Confusions that cross the letter/digit line** — `0`/`O`, `A`/`4`, `J`/`1`, `U`/`0` —
are correctable, because an Indian plate's format says which class belongs at each
position. Extending the substitution tables from this measurement raised exact reads
from **30.5 % to 35.2 %**.

**Confusions inside one class** — `4` for `1`, `7` for `2`, `M` for `H` — are not.
Knowing "a digit belongs here" cannot separate two digits. That is the larger half of
what remains, and no amount of format repair reaches it.

### Reading: solved by changing the recogniser, not the rules

The conclusion above — that the remaining half needed a recogniser trained on plates —
was right about the diagnosis and wrong about the cure. A plate-trained recogniser was
built here and reached 46.1%, better than EasyOCR but not enough. The answer turned out
to be simpler and was sitting in an already-installed library.

Every backend measured the same way, on the same 400 plates:

| Recogniser | Whole plate right | Characters right | Seconds a plate |
| --- | --- | --- | --- |
| EasyOCR *(what shipped before)* | 34.5 % | 71.2 % | 0.56 |
| A CRNN trained here on 1380 crops | 46.1 % | 75.1 % | — |
| PaddleOCR, full pipeline | 60.2 % | 80.3 % | 5.59 |
| **PaddleOCR, recogniser only** *(the default now)* | **73.2 %** | **90.7 %** | **0.22** |

**More than double the plates read exactly, at 2.5× the speed.** That combination looks
impossible until the reason is clear: **the plate is already located before any of this
runs.** PaddleOCR's full pipeline pays to detect text all over again, and its detector
splits one plate into fragments that then have to be stitched back together — which is
why the full pipeline is both slower *and* less accurate than its own recogniser used
alone.

What the shortcut gives up is row separation: a two-line plate arrives as one image with
nothing to mark where the first line ends. The 400 plates include two-line plates and it
still wins by a wide margin, so the trade is worth taking — but it is not free, and the
full pipeline is kept available for that reason.

```bash
python main.py --source clip.mp4 --detect all --ocr easyocr   # force the old one
python tools/plate_benchmark.py --engine paddle-rec --limit 400
```

The default picks the best recogniser that will actually load, so a machine without
PaddleOCR falls back to EasyOCR and reads plates worse rather than not at all. The
startup line names whichever one it got, so a run is never ambiguous about it.

### Repair has a limit, deliberately

Extending those tables also let *more wrong numbers* through the format check: 74 wrong
accepts became 92. A wrong plate is not a harmless miss — it reaches the vehicle
registry and flags an innocent driver.

So repair may change at most three characters. Past that it is not recovering a number,
it is manufacturing one:

| | exact | wrong accepts | precision |
| --- | --- | --- | --- |
| Before | 30.5 % | 74 | 61.3 % |
| Extended tables | 35.2 % | 92 | 59.6 % |
| **Extended, capped at 3 repairs** | **34.5 %** | **84** | **61.3 %** |

More plates read correctly, at exactly the precision it started with.

### What actually makes it safe: reading a plate more than once

Everything above scores a *single* read. The pipeline never acts on one. It gathers
reads across frames and votes per character position, reporting a number only when
enough of them agree — because real reads of one plate disagree in *different* places:

```
DL9CZ2581  +  DL5CZ2581  +  DL9CZ2581   ->  DL9CZ2581
```

The dataset contains several photographs of the same plate, so this can be measured
rather than assumed — each group stands in for the frames of one pass:

```bash
python tools/plate_benchmark.py --consensus 5 --min-agreeing 2
```

| 219 plates, up to 5 reads each | |
| --- | --- |
| Reported a number | 107 (48.9 %) |
| …and were right | 86 |
| …and were wrong | 21 |
| Said nothing | 112 |
| **Precision** | **80.4 %** |
| Recall | 39.3 % |

**Precision rises from 61 % to 80 % once a plate is read more than once**, and the
system stays silent on the half it is unsure of. That is the right shape for a border
post: a number you can act on, or an honest blank, and the crop saved either way so an
operator can read what the machine would not commit to.

It is still 1 wrong in 5 reported. That is not yet good enough to drive an automatic
registry check unattended, and the README will say so until the number moves.

### Training a recogniser: the wrong turn, kept because it was instructive

Same-class confusions cannot be repaired by format, and the conclusion drawn at the time
was that a recogniser trained on Indian plates was the only thing that would move them.
That was the wrong turn — the answer was a better general recogniser, already installed —
but the work is recorded rather than deleted, because it is what established that the
recogniser was the bottleneck and not the rules around it.

The labelled crops were cut from the annotated dataset:

```bash
python tools/plate_dataset.py
```

1691 crops — 1380 train, 311 validation — split **by plate number, not by photograph**.
One number appears in 41 photographs here, and letting it fall on both sides of the
split would let a model score well by memorising a car rather than reading a plate. The
split is verified to have zero overlap.

```bash
python tools/train_plate_reader.py --epochs 150
```

A CRNN read end to end with CTC — convolutions over the crop, a recurrent pass along its
width, no character segmentation — trained with the degradations that cause the real
failures: distance (downscale and back), blur, motion, glare, perspective, noise.

It starts at a large disadvantage, which is worth stating rather than discovering later:
**1380 crops against the millions EasyOCR was trained on.**

It reached **46.1% exact, 75.1% of characters** — clearly past EasyOCR's 34.5%, and
clearly short of the 73.2% that swapping the recogniser gave for no training at all. Two
things are worth taking from it:

* **Augmentation can destroy the signal it is meant to protect.** The first attempt
  applied every degradation at high probability and the model then could not fit its own
  training data — 12% exact, 49% of characters. Showing it an unaltered plate a third of
  the time fixed that. A model has to read a clean plate before it can read a ruined one.
* **It confirmed the diagnosis.** A small model on 1380 crops beating a general
  recogniser by 12 points established that the recogniser was the limit — which is what
  made trying a different one the obvious next move.

The run stopped at epoch 50 of 150 without an error and was not restarted, since the
recogniser swap had by then overtaken it. `models/plate-reader.pt` holds that epoch.

### Where it stands

* **Plate localisation** — works on large plates from the frame pass, and on small ones
  through the close-up rescue. Verified on real traffic.
* **Plate crop as evidence** — works, and is genuinely the plate.
* **Blur rejection, format validation, refusal to guess** — work.
* **Automatic number reading** — **works**: 73.2% of real Indian plates read exactly,
  90.7% of characters, and higher again across several frames of one pass.

What is still not measured is reading on *low-resolution* footage end to end. The 73.2%
is recognition measured on annotated crops; on 848×478 phone video the limit is finding a
plate with enough pixels on it, not reading one. Those are different problems and only
the second is solved.

---

## Face recognition — recognising who belongs

The purpose is **subtraction, not identification**. A guard walking his own patrol line
trips exactly the same virtual fence as somebody climbing over it, and an operator shown
both learns to ignore both. Recognising the guard is what lets the other alert mean
something.

```bash
python tools/fetch_face_models.py
python tools/enrol_face.py --id SSB-114 --label "Havildar Singh" --role "day patrol" photo1.jpg photo2.jpg
python main.py --source webcam:0 --faces config/face-roster.json
```

The roster file is written by the enrolment step; nothing ships one. A repo
carrying a pre-enrolled roster would be carrying people's biometrics into a git
history, which is not a thing to do by accident.

Detection and embedding both use OpenCV's own models (YuNet and SFace), so this adds no
Python dependency — only two model files.

### Two deliberate asymmetries

**Being recognised lowers risk by 45 points. Not being recognised raises nothing.**
Almost nobody at a border is enrolled and never will be, so treating an unknown face as
evidence would flag the entire population. Absence from a roster is the normal case, not
a signal. A checkpoint where everyone genuinely *should* be enrolled can turn it on with
`unrecognised_points`.

**One sighting is never an identity.** Two agreeing sightings are required before an
identity is acted on, for the same reason a single plate read is not trusted: acting on a
guess is how the wrong person gets waved through.

### It is a gate capability, not a perimeter one

A face is about one eighth of a person's height, so a person 200 px tall has a face of
roughly 25 px. Faces below 60 px are refused outright, because below that the vector is
noise and matching noise against a roster invents people.

```bash
python tools/face_envelope.py --photo somebody.jpg
```

That tool matches a face against *itself*, scaled down — the same pixels on both sides,
so the errors are correlated and the score is far kinder than reality. Read its floor as
a **ceiling on optimism**: failing there means certain failure in the field, while passing
says nothing about matching a photograph enrolled weeks earlier under different light.

The faces in the bundled sample photo are 38–42 px — below the working minimum, which is
itself the demonstration.

### Biometric data

Three decisions follow from what this stores, and they are deliberate:

* **Only vectors, never photographs.** A face vector cannot be turned back into a usable
  image. A test asserts the roster file contains nothing else.
* **Events carry the identity, not the biometrics** — the roster id, label and role the
  post chose, never the vector.
* **Enrolment is manual and explicit.** Nothing in the running pipeline ever adds a
  person it merely saw. A system that quietly builds a biometric database of passers-by
  is a different system than this one, and not one this project becomes by accident.

Who may lawfully be enrolled, and on what basis, is a question for the deploying
authority. This code cannot answer it and does not try.

---

## The pass log — every vehicle, not only the ones that alarm

Alerts answer *what went wrong*. A post also gets asked the other question: **what came
through here today?** Most vehicles break no rule, so an alert log never contains them —
and a week later that is exactly the list somebody wants.

So every vehicle is logged, in any zone, alarm or not, to `data/events/passes-YYYYMMDD.jsonl`:

```json
{"at": "2026-08-27T01:19:56", "camera_id": "CAM-01", "track_id": 2, "vehicle": "CAR",
 "plate": null, "plate_read": false, "seconds_in_view": 3.6, "frames": 104,
 "crop": "data/evidence/20260827-011956_CAM-01_pass-CAM-01-2.jpg"}
```

Two things make that list usable rather than a wall of noise, and they pull against
each other:

**One entry per vehicle, not per frame.** A vehicle sits in view for hundreds of frames.
On a real street clip, 1842 frames produced **48 entries** — one of them a car seen
across 187 consecutive frames.

**But a vehicle that comes back is a second entry.** A car passing at 09:00 and again at
09:02 is precisely the pattern worth seeing. Duplicates are suppressed only inside a
short window (`--plate-log-window`, 10 s by default); past it the same plate is a genuine
second crossing.

The entry is written when the vehicle **leaves**, not when it arrives, so the crop kept
is the sharpest of the whole pass rather than the first blurred glimpse. A short
occlusion — passing behind a bus — does not split one pass into two.

`plate_read` is in every row and is honest: on real footage it is almost always `false`,
because [reading still does not work](#anpr-an-honest-account). The crop is saved anyway,
because an operator can read a plate the machine would not commit to.

```bash
python main.py --source clip.mp4 --detect all              # on by default
python main.py --source clip.mp4 --detect all --no-plate-log
python main.py --source clip.mp4 --detect all --plate-log-window 60
```

---

## Where the cameras are — the map

An alert that says `CAM-05` means nothing to somebody who does not already know where
CAM-05 is. On a map it means *this stretch of the river*, and a commander can send the
nearest patrol instead of asking which camera that was.

Locations live beside the cameras they belong to, in the fleet file, so there is no
second file to keep in step:

```json
{
  "camera_id": "CAM-01",
  "source": "webcam:0",
  "location": { "lat": 28.613939, "lon": 77.209023,
                "bearing": 90, "fov": 60, "height_m": 1.2, "site": "Gate A" }
}
```

Everything here is **optional**. A camera without coordinates runs exactly as it does
now — it is simply listed as `unplaced` by the map API rather than dropped, so it is
visible that it exists and has not been surveyed yet.

```
GET /api/map                      every camera's position, wedge and areas
GET /api/map/cameras/CAM-01       one camera, plus its nearest neighbours in metres
```

`nearest` answers the question a map actually gets asked during an incident: *somebody
left CAM-02 heading east — which camera should be watched now?*

### Two distinctions that matter

**A zone is not a geofence.** The virtual fences in `config/zones-*.json` live in *frame*
coordinates — fractions of the image, drawn by eye on what one camera sees. They cannot
be placed on a map, because converting image coordinates to ground coordinates needs a
surveyed homography per camera that most posts will not have. A geofence here is the
opposite: a polygon in latitude and longitude that knows nothing about any camera's view.
Both are useful; neither converts into the other for free.

**A bearing wedge is aim, not coverage.** `bearing` plus `fov` draws a wedge showing
which way a camera points. A ridge, a wall or a parked truck takes most of that away. The
wedge is only drawn when both are actually surveyed — a wedge from a guessed bearing
points confidently at the wrong hillside, which is worse than no wedge at all.

Nothing in the running pipeline tests detections against geofences yet, and the README
will say so until it does. This is the backend a map view will sit on, built now so the
data model does not have to be retrofitted later.

---

## Vehicle registry

IBVAP checks a read plate against what is registered to it, and flags a mismatch — a
plate registered to a motorcycle, bolted to a truck, is the signature of a cloned or
transplanted plate.

> **Read this next to [ANPR: an honest account](#anpr-an-honest-account).** Everything
> here begins with *a read plate*, so `PLATE MISMATCH`, `WATCHLISTED VEHICLE` and
> `UNREGISTERED PLATE` are only ever as good as the read beneath them. That read is now
> 73.2% exact on annotated crops and better again across several frames of one pass — but
> on low-resolution footage the limit is finding a plate with enough pixels on it at all,
> which is a separate and unfinished problem. Treat a registry hit as evidence to look
> at, not as a fact to act on unattended.

**This does not talk to VAHAN or any RTO system.** It reads a local registry file, which
is what a border post would actually be issued: a periodic extract or a watchlist.
`RegistryBackend` is the seam an authorised API would slot into later; nothing above it
would change. `config/vehicle-registry.json` holds invented sample data.

Registry extracts carry owner details. Only the vehicle attributes needed for the
comparison are copied into an event — owner information stays in the registry and out of
the event log, which a test enforces.

---

## Following one person across cameras

A vehicle carries its identity on the outside: read the plate on CAM-01 and again on
CAM-03 and you know it is the same vehicle. A person carries nothing. All that is left is
appearance, which is a far weaker signal — the same coat is a different colour under
sodium light than under infrared, and two guards in one uniform look alike at any
distance.

So the fleet links people conservatively, and says plainly that it is guessing:

* **A much higher bar than within-camera re-identification** (0.72 against 0.55).
  Recovering a track through a two-second occlusion is a different problem from claiming
  a match across two cameras minutes apart.
* **Overlapping sightings are never linked.** If CAM-01 can still see them while CAM-03
  starts to, that is two people — unless a post declares its cameras genuinely overlap.
* **A link needs somewhere plausible to have walked from.** Beyond the travel window it
  is a different person who happens to own a similar coat.
* **Every cross-camera subject is marked `inferred`**, and carries which cameras actually
  saw what. A movement history has to distinguish an observation from a deduction.

It answers "has this person been seen elsewhere on this post?" It is **not
identification** and must never be presented as such — that is what the face roster is
for, and only at a gate.

---

## Evidence

Each alert produces up to three artefacts sharing one filename stem:

| File | Contents |
| --- | --- |
| `..._<id>.jpg` | The annotated frame: zones, boxes, ground points, HUD |
| `..._<id>.mp4` | Footage from a few seconds before the alert to a few after |
| `..._<id>-plate.jpg` | The plate crop, when a vehicle was involved |

The pre-roll is the part that cannot be captured any other way: by the time an alert
exists, the approach has already happened. Every frame goes into a small rolling buffer
so the run-up is already in hand. Clips are raw footage, not overlaid — evidence should
be what the camera saw.

Alerts also go to an append-only JSONL log and, with `--db`, to a queryable SQLite store
that supports lookup by camera, severity, time and plate.

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

926 tests, none of which need a model, a camera or a network. Highlights of what they
pin down rather than merely cover:

* A person standing exactly on a fence line counts as inside it.
* Loitering survives an occlusion, because re-identification recovers the track.
* Two camera threads never both touch OpenCV's GUI — that combination deadlocked.
* A stateful detector is never shared between cameras; ByteTrack would mix up track IDs.
* A blurred plate is skipped rather than guessed at.
* A plate handed from one camera to another is labelled as inferred, with the camera
  that actually read it.
* An event's risk score always equals the sum of its printed factors.

---

## Detection envelope — how far a camera actually sees

A model does not care about metres, it cares about pixels. Before mounting anything it is
worth knowing how small a person can get before detection stops:

```bash
python tools/detection_envelope.py
```

Measured on YOLOv8n at `imgsz 640`, three runs per height:

| Person height in frame | Daylight | Night simulation |
| --- | --- | --- |
| 220 px and above | 100 % · conf 0.85 | 100 % · conf 0.76 |
| 160 px | 100 % · conf 0.85 | **33 %** · conf 0.52 |
| 120 px | 100 % · conf 0.79 | **0 %** |
| 90 px and below | **0 %** | **0 %** |

**Night costs roughly double the pixels.** Reliable to 120 px by day, but only to 220 px
under the night simulation. Size a lens against the night figure, not the day one.

To turn that into a range, use the camera's own field of view:

```
person height in pixels = frame height × 1.7 m ÷ (2 × range × tan(vFOV ÷ 2))
```

Two honest limits. The night column is a **simulation** — monochrome, sensor grain and
motion blur applied to daylight footage — and real infrared differs in ways that cannot
be faked, because IR reflectance is not visible reflectance. And it samples one person
against one background, so the numbers are an order of magnitude, not a specification.
Re-run it against your own footage and camera before trusting it.

---

## Performance — where the frames go

Profiled rather than guessed, on a CPU-only laptop against the bundled vehicle clip:

| Stage | Share of wall time |
| --- | --- |
| ANPR (plate model + OCR) | **60 %** |
| YOLO detection | ~36 % |
| Clip buffer | 2 % |
| Tracking, context, behaviour, risk, zones, registry | **under 1 % combined** |

The whole analysis stack — every behaviour, every zone test, every risk calculation — is
free. Only the two neural stages cost anything.

### The one free win: input size

`imgsz` is the single biggest lever, and 640 was not the right default:

| imgsz | ms/frame | Smallest person seen, day | …at night |
| --- | --- | --- | --- |
| 960 | 165 | 55 px | 220 px |
| 640 | 108 | 120 px | 220 px |
| **512** | **73** | **120 px** | **220 px** |
| 416 | 80 | 160 px | 220 px |
| 320 | 51 | 220 px | 300 px |

**512 is 32 % faster than 640 and sees exactly as far.** That is now the default, and it
lifted the pipeline from 8.8 to 10.3 fps with no detection loss. Below 512 the trade
becomes real — 320 is faster again but stops seeing anyone under 220 px in daylight.

```bash
python tools/detection_envelope.py --sweep
```

Re-run that against your own camera and range before changing it. The figures move a
little between runs, because detection near a floor is marginal by definition.

### How many cameras one box can carry

Measured with the fleet runner, one shared model, person detection only:

| Cameras | Total fps | Per camera |
| --- | --- | --- |
| 1 | 13.3 | 13.3 |
| 2 | 11.6 | 5.8 |
| 4 | 10.9 | 2.7 |
| 6 | 10.9 | 1.8 |
| 8 | 10.7 | **1.3** |

**Total throughput is flat at about 11 fps whatever the camera count.** The box
saturates; per-camera frame rate is simply 11 ÷ n. Two cameras is comfortable, four is
marginal, eight is not surveillance.

That last row carries a correctness consequence, not just a slow one. `confirm_frames`
defaults to 3, which is a fifth of a second on one camera and **2.3 seconds** on eight —
long enough for someone to cross a zone between the frames that would have confirmed
them. The pipeline now measures its own frame rate and says so:

```
[WARN] CAM-01 1.3 fps means 3 confirmation frames take 2.3s - a running person
       can cross a zone in that time. Lower --confirm-frames or run fewer
       cameras. Do NOT raise --stride: it analyses fewer frames, which makes
       this window longer, not shorter.
```

### ANPR is expensive, and threading does not fix it

Reading one plate costs about **1.4 seconds** — the recogniser, not the plate model,
which is 199 ms. Two or three agreeing reads are needed before a number is trusted, so
roughly 4 seconds per vehicle. Enabling ANPR takes the pipeline from 8.8 fps to 5.4.

Moving that work to a background thread was tried, and **measured worse**: 5.4 fps →
4.7. Threads do not create cores. Detection and recognition end up fighting over the same
ones, and the frame copies handed to the worker cost more on top. On a machine with a GPU
the two would use different silicon and the picture reverses, so the option survives as
`--anpr-thread`, off by default.

Throttling did not help either: the read already stops after three agreeing attempts, so
the interval was never the binding constraint.

**The honest position:** on CPU, ANPR costs what it costs. Run `--no-anpr` for full frame
rate, or put detection on a GPU.

### What `--stride` does, and does not, do

It was recommended here as a throughput mitigation before it was measured. It is not one:

| stride | Frames analysed | Rate | Footage covered |
| --- | --- | --- | --- |
| 1 | 72 | 14.0 fps | 100 % |
| 2 | 36 | 13.5 fps | 50 % |
| 3 | 24 | 11.2 fps | 33 % |

**The processing rate does not change.** Stride analyses *fewer frames*, so a file
finishes sooner — but on a live feed it means looking at less of what the camera saw.

It also makes the confirmation window **worse**, not better: three analysed frames cover
twice as much wall-clock time at stride 2, which is the opposite of what a slow camera
needs. On an RTSP feed it is close to redundant anyway, because the capture buffer is
already pinned to one frame and drops stale ones on its own.

Use it to cut load when covering less footage is genuinely acceptable — a slow-moving
scene, a wide overview camera. Not to make a fence react faster.

---

## Implemented now / Next / Future

The original brief asked for this breakdown by name. It is worth reading alongside the
note below it, because the project has since travelled a long way past that brief.

### IMPLEMENTED NOW

* Person and vehicle detection (YOLOv8) on image, video, webcam, RTSP and HTTP sources
* IoU tracking with appearance re-identification through occlusions
* Configurable polygon virtual fences, in normalized coordinates, with per-object rules
* Context Engine: speed, heading, dwell, approach, apparent size, heading variance
* Behaviour Engine: loitering, border-facing, erratic, night, rapid, camera tampering
* Camera integrity: link loss treated as a possible tampering event
* Explainable 0–100 risk scoring where every point traces to a named factor
* Alerting with confirmation frames, cooldown, and escalation that overrides both
* Evidence: annotated still, video clip with pre-roll, plate crop
* JSONL audit log and a queryable SQLite event store, with incident grouping
* Optional ground-plane calibration (speed in m/s, distances in metres)
* ANPR: plate localisation and evidence capture — **not** automatic number reading
* Vehicle registry check: a plate that does not match its vehicle
* Face recognition against a roster of expected people (gate range only)
* Cross-camera linking: vehicles by plate, people by appearance
* Multi-camera fleet in one process, sharing one loaded model
* Dashboard, REST API and live WebSocket alert feed
* 926 automated tests, none needing a model, a camera or a network

### NEXT

* **A recogniser trained on Indian plates.** The one gap that is measured, understood
  and unfixable from here — see [ANPR](#anpr-an-honest-account).
* **Real night and IR footage.** Everything about night performance rests on a
  simulation, and simulation cannot answer how infrared reflectance behaves.
* **Zones specified in ground metres** rather than frame coordinates, now that a
  homography exists — moving a camera currently means redrawing its fences.
* **A GPU path.** `--device` exists and has never been run on one; it is also the only
  route past ~11 fps per box.

### FUTURE

* Abnormal-behaviour detection against a learned per-camera baseline of "normal"
* Learned rather than hand-tuned risk weights
* An event store shared across posts, and integration with existing control rooms
* Edge deployment on the camera side rather than a central box
* Group behaviour: several tracks converging, or moving in formation

---

## How far this has travelled from the original brief

The brief that started this asked for a small MVP — person detection plus a virtual
fence — and listed things **not** to build yet: a dashboard, facial recognition, ANPR,
multi-camera tracking, a database.

Every one of those now exists, because they were asked for afterwards, one at a time.
That is worth stating plainly rather than leaving someone to notice the contradiction:
the "do not build" list in the original brief is superseded, not forgotten.

What has *not* changed is the discipline the brief set — nothing is claimed as working
that has not been run. Where a feature only half works, the README says which half.

---

## Known weaknesses

1. **Zones are still image-space polygons.** Calibration gives real distances *to* a
   zone, but the fence itself is drawn in frame coordinates, so moving a camera means
   redrawing it.
2. **Track identity is the weak link under everything.** Every behaviour rests on it.
   Appearance re-ID survives a few seconds of occlusion; it will not tell two people in
   the same uniform apart.
3. **ANPR needs a plate-specific model** — see the honest account above.
4. **Night performance is measured only against a simulation.** Detection survives
   monochrome conversion, heavy grain and motion blur, but needs roughly double the
   pixels — see [Detection envelope](#detection-envelope--how-far-a-camera-actually-sees).
   What no simulation can answer is infrared *reflectance*: clothing that is dark to the
   eye can be bright under IR and the reverse, and that is a property of the sensor and
   the scene, not of the image. Real SSB night footage is the only way to settle it, and
   a larger or fine-tuned model is the likely answer if it falls short.
5. **Calibration assumes one flat ground plane.** A person on a slope or a vehicle roof
   will have their distance and speed misreported, and accuracy degrades toward the
   horizon.
6. **Throughput.** About 10 fps per stream on a CPU-only laptop, or 5 with ANPR on —
   see [Performance](#performance--where-the-frames-go). Below real-time for one camera,
   let alone many. A GPU is the real answer; `--no-anpr` is the mitigation available
   today. `--stride` is not one - it analyses less footage at the same rate.
7. **Behaviour thresholds are policy, not science.** Every number was tuned against demo
   footage, not learned from border data. They are all constructor arguments so they can
   be retuned per deployment, and they will need to be.

---

## Not built

React frontend, PostgreSQL, cloud or Kubernetes deployment, learned anomaly detection,
and authentication. These are deliberate omissions, not oversights.

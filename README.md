# cv_tasks — OpenCV / YOLO computer vision tasks on ROS Noetic

Five computer vision case studies, each solved in Python and wired into ROS Noetic
with the node structure required by the module:

```
roscore (master)
   └── video_publisher.py  ── /video/frame (sensor_msgs/Image) ──►  <task>_subscriber.py
        cv2.VideoCapture()                                           runs the solution + shows the result
```

| # | Task | Method | Subscriber node |
|---|------|--------|-----------------|
| 1 | Detect triangle / square / pentagon / hexagon / circle (background color changes) | per-frame background estimation → contours → circle-fill ratio + `approxPolyDP` | `shape_subscriber.py` |
| 2 | Unique colors + segment the most frequent color (per pixel) | quantized 3D histogram + Lab merge of similar bins | `color_subscriber.py` |
| 3 | Detect all falling diamonds (no classification) | adaptive threshold on brightest channel + morphology + external contours | `diamond_subscriber.py` |
| 4 | Remove noise from a noisy mask video (white donut) | Otsu + connected components + small-hole fill + median blur | `denoise_subscriber.py` |
| 5 | Find Hollow Knight in a video | YOLO (`.onnx` via `cv2.dnn`, or `.pt` via ultralytics) | `hollow_knight_subscriber.py` |

## Layout

```
cv_tasks_ws/                    <- repo root = catkin workspace
└── src/cv_tasks/
    ├── src/cv_tasks/           pure OpenCV/NumPy solutions (no ROS imports, unit-tested)
    ├── scripts/                ROS nodes: one publisher + one thin subscriber per task
    ├── launch/                 one launch file per task (publisher + subscriber together)
    ├── models/                 put best.onnx / best.pt here (git-ignored)
    ├── data/                   put the videos here (git-ignored)
    └── test/test_library.py    synthetic-frame tests for the library
```

## Requirements

- Ubuntu 20.04 + ROS Noetic (`ros-noetic-cv-bridge`), Python 3
- `python3-opencv`, `python3-numpy` (installed with ROS desktop / cv_bridge)
- Task 5 with a `.pt` model only: `pip3 install ultralytics`

## Build

```bash
git clone <this repo> cv_tasks_ws && cd cv_tasks_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Run

Easiest — one command per task (starts `roscore` automatically):

```bash
roslaunch cv_tasks shape.launch   video:=$HOME/videos/shapes.mp4
roslaunch cv_tasks color.launch   video:=$HOME/videos/cartoon.mp4
roslaunch cv_tasks diamond.launch video:=$HOME/videos/diamonds.mp4
roslaunch cv_tasks denoise.launch video:=$HOME/videos/donut.mp4
roslaunch cv_tasks hollow_knight.launch video:=$HOME/videos/hk.mp4 \
          model:=$(rospack find cv_tasks)/models/best.onnx imgsz:=1280
```

Useful launch arguments: `save_to:=results/out.mp4` (record the result), `show:=false`
(headless), `loop:=true`, `fps:=10` (throttle the publisher for slow models),
`proc_width:=` (analysis resolution, tasks 2-4), `cuda:=true` (task 5).

Manual way, three terminals:

```bash
roscore
rosrun cv_tasks video_publisher.py _video:=/path/to/video.mp4
rosrun cv_tasks diamond_subscriber.py
```

Press `q` or `Esc` in the preview window to stop a subscriber. On shutdown every
subscriber prints how many frames it processed and the average FPS.

## Task 5 notes

- Export a trained model for OpenCV: `yolo export model=best.pt format=onnx imgsz=1280`
  and pass the same `imgsz:=1280` when running.
- The character is small in the frame (~4% of its width), so 960–1280 usually beats 640.
- The publisher keeps only the newest frame (`queue_size=1`), so a slow model skips frames
  instead of lagging. Use `fps:=` to slow the publisher if every frame must be processed.

## Tests (no ROS needed)

```bash
cd src/cv_tasks && python3 -m unittest discover -s test -v
```

## Credits / sources

Fill in before submitting (module rule 4 — academic integrity):

- Task 1: synthetic shapes video (source: ...)
- Task 2: animation by [2veinte](https://2veinte.com.ar/)
- Task 3: "Diamonds jewels rain falling" by Vimeo-Free-Videos, [Pixabay](https://pixabay.com/videos/diamonds-jewels-rain-falling-3122/)
- Task 4: [Pink donut floating in mid air with alpha channel, Getty Images](https://www.gettyimages.com/detail/video/pink-donut-floating-in-mid-air-with-alpha-channel-stock-footage/1255690170)
- Task 5: video by Qwarky (YouTube); dataset: Hollow Knight Dataset, modified from the Hollow Knight Computer Vision Model
- All algorithms are original work using OpenCV / NumPy; YOLO via ultralytics where used.

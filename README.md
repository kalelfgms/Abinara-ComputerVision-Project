# OPENCV Tasks
Pada KPP ini diberi 5 task computer vision yang harus diselesaikan dengan
Python, tiap task diberi video yang harus diolah dan memakai metode yang berbeda sesuai masalahnya; OpenCV untuk task 1-4,
dan model YOLO untuk task 5.

Pengerjaan diintegrasikan dengan ROS Noetic: satu publisher membaca video dan
kirim tiap frame ke topic `/video/frame`, lalu subscriber tiap task
menerima frame itu, mengolah dan menjalankan kodenya, dan menampilkan hasilnya.

Dengan launch file, tiap task bisa dirun dengan 1 command `roslaunch`.

## Files

### Supporting Files
| File | Function |
|------|--------|
| `setup.py` | Membuat package `cv_tasks` bisa diimport oleh ROS |
| `common.py` | Fungsi bantu yang dipakai di seluruh file: `FpsMeter` (hitung FPS) dan `put_label` (tulis teks di gambar) |
| `ros_node.py` | Kode dasar semua subscriber: menerima frame, menjalankan task, menampilkan window, gambar FPS, menyimpan video, dan berhenti saat video selesai |
| `video_publisher.py` | Node publisher: membaca file video (path dari argumen launch) dengan `cv2.VideoCapture()`, lalu mengirim tiap frame ke rostopic `/video/frame` |

### File per task
Tiap task punya 3 file agar bisa di run dengan ROS.  
Contoh file-file Task 1 (shape detection):

| File | Function |
|------|----------|
| `shape_detection.py` | Program utama task untuk mengolah video menggunakan OpenCV/NumPy |
| `shape_subscriber.py` | Node subscriber: terima frame dari `/video/frame`, memanggil program utama, menampilkan hasil |
| `shape.launch` | Menjalankan publisher + subscriber dengan 1 command `roslaunch` dan bisa ubah parameter dengan argumen |

### All task files
| Task | Library  | Subscriber |	Launch |
| ---- | -------  | ---------- | ------ |
| Deteksi bentuk  | `shape_detection.py`   | `shape_subscriber.py`         | shape.launch
| Warna dominan	  | `dominant_color.py`	   | `color_subscriber.py`         | color.launch
| Deteksi berlian	| `diamond_detection.py` | `diamond_subscriber.py`       | diamond.launch
| Hapus noise	    | `mask_denoise.py`	     | `denoise_subscriber.py`	     | denoise.launch
| Hollow Knight	  | `hollow_knight.py`     | `hollow_knight_subscriber.py` | hollow_knight.launch

## Workspace Structure
```
├── README.md
├── .gitignore
├── build, devel               // hasil catkin_make
└── src
    └── cv_tasks
        ├── CMakeLists.txt
        ├── package.xml
        ├── setup.py
        ├── data               // video tiap task 
        ├── models             // model YOLO hasil training untuk task #5
        ├── launch             // launch file tiap task
        ├── scripts            // node ROS: video_publisher.py + 1 subscriber per task
        └── src
            └── cv_tasks       // logika OpenCV tiap task + file pendukung
                ├── shape_detection.py
                ├── dominant_color.py
                ├── diamond_detection.py
                ├── mask_denoise.py
                ├── hollow_knight.py 
                ├── common.py  
                └── ros_node.py
```

## Tasks & Terminal Commands
**Note:** `roslaunch` otomatis menjalankan `roscore` jika belum berjalan,
### WHERE ARE THE SHAPES?
```
roslaunch cv_tasks shape.launch video:=$(rospack find cv_tasks)/data/shape_detection_static.mp4
```
or
```
roslaunch cv_tasks shape.launch video:=$(rospack find cv_tasks)/data/shape_detection_transition.mp4
```

### HOW COLORFUL?
```
roslaunch cv_tasks color.launch video:=$(rospack find cv_tasks)/data/lucky_charms.mp4
```

### LOCATE IT, PLEASE
```
roslaunch cv_tasks diamond.launch video:=$(rospack find cv_tasks)/data/diamonds.mp4
```

### I DONT THINK ITS RELIABLE
```
roslaunch cv_tasks denoise.launch video:=$(rospack find cv_tasks)/data/noisy_mask.mp4
```

### DOES IT MATCH?
```
roslaunch cv_tasks hollow_knight.launch \
  video:=$(rospack find cv_tasks)/data/gameplay_2min.mp4 \
  model:=$(rospack find cv_tasks)/models/best.onnx imgsz:=640
```

## Documentation
[Google Drive KPP 5 Internship Abinara-1](https://drive.google.com/drive/folders/1lP6cTCHHoke5eX4u7nFPINA2VyA8qYaQ?usp=sharing)

## Credits & Sources

### Code
- The OpenCV/NumPy solutions and the ROS node structure were developed with
  Claude as an AI coding assistant, then ran and tested by me.
- Libraries: OpenCV, NumPy, ROS Noetic (`rospy`, `cv_bridge`), Ultralytics YOLO.

### Model Training
- Hollow Knight detector: YOLO11n trained with Ultralytics in
  [Google Colab](https://colab.research.google.com/drive/1mk6euC4ydz_2ZMXnlMJnc_8R2SbW8Ekx?usp=sharing)

### Videos
- Task 1: provided by abinara-1 team
- Task 2: animation by [2veinte](https://2veinte.com.ar/)
- Task 3: "Diamonds jewels rain falling" by Vimeo, [Pixabay](https://pixabay.com/videos/diamonds-jewels-rain-falling-3122/)
- Task 4: [Pink donut floating in mid air with alpha channel, Getty Images](https://www.gettyimages.com/detail/video/pink-donut-floating-in-mid-air-with-alpha-channel-stock-footage/1255690170)
- Task 5: gameplay video by Qwarky (YouTube)

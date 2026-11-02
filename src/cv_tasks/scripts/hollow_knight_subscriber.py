#!/usr/bin/env python3
"""Task 5 - Hollow Knight detection with YOLO.

Parameters:
    ~model   path to best.onnx / best.pt (required)
    ~imgsz   inference size, must equal the export size for ONNX (default 640)
    ~conf    confidence threshold (default 0.4)
    ~iou     NMS IoU threshold (default 0.5)
    ~names   comma-separated class names, ONNX only (default hollow_knight)
    ~cuda    use the GPU (default false)
"""

import sys

import rospy

from cv_tasks.hollow_knight import draw, load_detector
from cv_tasks.ros_node import VideoNode


def main():
    node = VideoNode("hollow_knight_subscriber")

    if not rospy.has_param("~model"):
        rospy.logfatal("missing parameter ~model  (e.g. _model:=/path/to/best.onnx)")
        sys.exit(1)

    detector = load_detector(
        rospy.get_param("~model"),
        imgsz=int(rospy.get_param("~imgsz", 640)),
        conf=float(rospy.get_param("~conf", 0.4)),
        iou=float(rospy.get_param("~iou", 0.5)),
        names=str(rospy.get_param("~names", "hollow_knight")).split(","),
        cuda=bool(rospy.get_param("~cuda", False)),
    )
    names = detector.names if isinstance(detector.names, dict) else dict(enumerate(detector.names))

    def process(frame):
        return draw(frame, detector(frame), names)

    node.run(process)


if __name__ == "__main__":
    main()

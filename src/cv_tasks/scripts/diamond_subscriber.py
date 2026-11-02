#!/usr/bin/env python3
"""Task 3 - object detection (falling diamonds, no classification).
Parameter: ~proc_width (analysis width, default 960, 0 = full resolution)."""

import rospy

from cv_tasks.diamond_detection import detect, draw
from cv_tasks.ros_node import VideoNode


def main():
    node = VideoNode("diamond_subscriber")
    proc_width = rospy.get_param("~proc_width", 960)

    def process(frame):
        boxes, _ = detect(frame, proc_width)
        return draw(frame, boxes)

    node.run(process)


if __name__ == "__main__":
    main()

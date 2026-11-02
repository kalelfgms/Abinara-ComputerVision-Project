#!/usr/bin/env python3
"""Task 2 - unique colors + dominant color segmentation.
Parameter: ~proc_width (analysis width, default 640, 0 = full resolution)."""

import rospy

from cv_tasks.dominant_color import process as run_task
from cv_tasks.ros_node import VideoNode


def main():
    node = VideoNode("color_subscriber")
    proc_width = rospy.get_param("~proc_width", 640)
    node.run(lambda frame: run_task(frame, proc_width))


if __name__ == "__main__":
    main()

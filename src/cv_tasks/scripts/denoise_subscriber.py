#!/usr/bin/env python3
"""Task 4 - noise removal on the noisy mask video.
Parameter: ~proc_width (processing width, default 0 = full resolution)."""

import rospy

from cv_tasks.mask_denoise import process as run_task
from cv_tasks.ros_node import VideoNode


def main():
    node = VideoNode("denoise_subscriber")
    proc_width = rospy.get_param("~proc_width", 0)
    node.run(lambda frame: run_task(frame, proc_width))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Task 1 - shape detection (triangle, square, pentagon, hexagon, circle)."""

from cv_tasks.ros_node import VideoNode
from cv_tasks.shape_detection import detect, draw


def main():
    node = VideoNode("shape_subscriber")

    def process(frame):
        results, _ = detect(frame)
        return draw(frame, results)

    node.run(process)


if __name__ == "__main__":
    main()

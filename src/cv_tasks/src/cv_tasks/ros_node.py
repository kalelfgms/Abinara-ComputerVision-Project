"""Shared plumbing for the subscriber nodes: receive a frame, run the task's
`process(frame) -> image`, overlay FPS, show it and optionally save it."""

import time

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from .common import FpsMeter, put_label


class VideoNode:
    """Private parameters (set with <param> in a launch file or _name:=value):
        ~topic    input topic            (default /video/frame)
        ~show     open a preview window  (default true)
        ~save_to  write result video     (default "" = don't save)
        ~save_fps fps of the saved video (default 30)
    """

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.bridge = CvBridge()
        self.topic = rospy.get_param("~topic", "/video/frame")
        self.show = rospy.get_param("~show", True)
        self.save_to = rospy.get_param("~save_to", "")
        self.save_fps = rospy.get_param("~save_fps", 30)
        self.writer = None
        self.meter = FpsMeter()
        self.process = None
        rospy.on_shutdown(self._on_shutdown)

    def run(self, process):
        """process(frame_bgr) -> image to display (BGR)."""
        self.process = process
        rospy.Subscriber(self.topic, Image, self._callback, queue_size=1, buff_size=2 ** 26)
        rospy.loginfo("%s listening on %s", self.name, self.topic)
        rospy.spin()

    def _callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        t0 = time.time()
        out = self.process(frame)
        ms = (time.time() - t0) * 1000.0

        fps = self.meter.tick()
        label = f"FPS {fps:.1f} | {ms:.0f} ms/frame"
        scale = 1.0 if out.shape[1] > 1000 else 0.6
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        put_label(out, label, org=(out.shape[1] - tw - 15, out.shape[0] - 15), scale=scale)

        if self.save_to:
            if self.writer is None:
                h, w = out.shape[:2]
                self.writer = cv2.VideoWriter(self.save_to, cv2.VideoWriter_fourcc(*"mp4v"),
                                              self.save_fps, (w, h))
                rospy.loginfo("saving result video to %s", self.save_to)
            self.writer.write(out)

        if self.show:
            cv2.imshow(self.name, out)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                rospy.signal_shutdown("quit key pressed")

    def _on_shutdown(self):
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        rospy.loginfo("%s: %s", self.name, self.meter.summary())

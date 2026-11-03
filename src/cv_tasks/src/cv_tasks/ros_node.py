"""Shared plumbing for the subscriber nodes: receive a frame, run the task's
`process(frame) -> image`, overlay FPS, show it and optionally save it.

All OpenCV window calls happen in the MAIN thread (the ROS callback only stores
the newest message), so the preview window never freezes, and the node shuts
itself down when the video has ended."""

import threading
import time

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from .common import FpsMeter, put_label


class VideoNode:
    """Private parameters (set with <param> in a launch file or _name:=value):
        ~topic        input topic                  (default /video/frame)
        ~show         open a preview window        (default true)
        ~save_to      write result video           (default "" = don't save)
        ~save_fps     fps of the saved video       (default 30)
        ~window_width preview window width in px   (default 960; window is resizable)
        ~idle_timeout seconds without frames before the node exits, i.e. the
                      video has ended (default 5, 0 = never exit by itself)
    """

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.bridge = CvBridge()
        self.topic = rospy.get_param("~topic", "/video/frame")
        self.show = rospy.get_param("~show", True)
        self.save_to = rospy.get_param("~save_to", "")
        self.save_fps = rospy.get_param("~save_fps", 30)
        self.idle_timeout = float(rospy.get_param("~idle_timeout", 5.0))
        self.window_width = int(rospy.get_param("~window_width", 960))
        self.writer = None
        self.meter = FpsMeter()
        self.process = None
        self._lock = threading.Lock()
        self._latest = None      # newest unprocessed message
        self._last_rx = None     # time of the last received message
        self._shown = False      # preview window has been created
        rospy.on_shutdown(self._on_shutdown)

    # ROS thread: just remember the newest frame
    def _on_msg(self, msg):
        with self._lock:
            self._latest = msg
            self._last_rx = time.time()

    def _window_closed(self):
        """True once the user closed the preview window with the X button."""
        if not (self.show and self._shown):
            return False
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def run(self, process):
        """process(frame_bgr) -> image to display (BGR)."""
        self.process = process
        rospy.Subscriber(self.topic, Image, self._on_msg, queue_size=1, buff_size=2 ** 26)
        rospy.loginfo("%s listening on %s", self.name, self.topic)

        # main thread: process + draw, keep the window responsive
        while not rospy.is_shutdown():
            with self._lock:
                msg, self._latest = self._latest, None
                last_rx = self._last_rx

            if msg is not None:
                self._handle(msg)
                continue

            if self.show:
                if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                    rospy.signal_shutdown("quit key pressed")
                elif self._window_closed():
                    rospy.signal_shutdown("window closed")
            else:
                time.sleep(0.02)

            if (self.idle_timeout > 0 and last_rx is not None
                    and time.time() - last_rx > self.idle_timeout):
                rospy.loginfo("no frames for %.0f s: video ended, shutting down", self.idle_timeout)
                rospy.signal_shutdown("video ended")

    def _handle(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        t0 = time.time()
        try:
            out = self.process(frame)
        except Exception as e:  # noqa: BLE001 - report once, then stop cleanly
            rospy.logerr("%s: processing failed: %s", self.name, e)
            rospy.signal_shutdown("processing error")
            return
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
            if not self._shown:
                cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)  # resizable by dragging
                h, w = out.shape[:2]
                cv2.resizeWindow(self.name, self.window_width, int(self.window_width * h / w))
            cv2.imshow(self.name, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                rospy.signal_shutdown("quit key pressed")
            elif self._shown and self._window_closed():
                rospy.signal_shutdown("window closed")
            self._shown = True

    def _on_shutdown(self):
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        rospy.loginfo("%s: %s", self.name, self.meter.summary())
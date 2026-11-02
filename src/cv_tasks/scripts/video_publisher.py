#!/usr/bin/env python3
"""Publisher: reads a video with cv2.VideoCapture and publishes every frame.

Private parameters:
    ~video               path to the video (required)
    ~topic               output topic (default /video/frame)
    ~fps                 publish rate; 0 = use the video's own fps (default 0)
    ~loop                restart the video when it ends (default false)
    ~wait_for_subscriber wait until someone subscribes before sending (default true)
"""

import sys

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main():
    rospy.init_node("video_publisher")

    if not rospy.has_param("~video"):
        rospy.logfatal("missing parameter ~video  (e.g. _video:=/path/to/video.mp4)")
        sys.exit(1)
    path = rospy.get_param("~video")
    topic = rospy.get_param("~topic", "/video/frame")
    fps_override = float(rospy.get_param("~fps", 0))
    loop = rospy.get_param("~loop", False)
    wait = rospy.get_param("~wait_for_subscriber", True)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        rospy.logfatal("cannot open video: %s", path)
        sys.exit(1)

    fps = fps_override if fps_override > 0 else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rospy.loginfo("publishing %s (%d frames) at %.1f fps on %s", path, total, fps, topic)

    pub = rospy.Publisher(topic, Image, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(fps)

    if wait:
        rospy.loginfo("waiting for a subscriber ...")
        while pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.sleep(0.5)  # let the connection settle so the first frames are not lost

    sent = 0
    while not rospy.is_shutdown():
        ok, frame = cap.read()
        if not ok:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        sent += 1
        rate.sleep()

    cap.release()
    rospy.loginfo("end of video, %d frames published", sent)


if __name__ == "__main__":
    main()

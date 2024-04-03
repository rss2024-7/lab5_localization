from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray

from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

import numpy as np

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.particle_positions = np.array([])
        self.laser_ranges = np.array([])
        self.particle_samples_indices = np.array([])

    def publish_avg_pose(self):
        particle_samples = self.particle_positions
        if len(self.particle_samples_indices) != 0:
            particle_samples = self.particle_positions[self.particle_samples_indices, :]

        positions = particle_samples[:, :2]
        angles = particle_samples[:, 2]

        average_position = np.mean(positions, axis=0)
        average_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        
        average_pose = np.hstack((average_position, average_angle))

        msg = Odometry()

        msg.header.frame_id = '/map'
        
        msg.pose.pose.position.x = average_pose[0]
        msg.pose.pose.position.y = average_pose[1]

        # rotation is around z-axis
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = np.sin(average_angle / 2)
        msg.pose.pose.orientation.w = np.cos(average_angle / 2)

        msg.child_frame_id = '/base_link'
        # self.get_logger().info("%s" % average_pose)
        # self.get_logger().info("%s" % average_angle)
        self.odom_pub.publish(msg)

    def odom_callback(self, msg):
        if len(self.particle_positions) == 0: return

        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        angle = msg.twist.twist.angular.z

        odometry = np.array([x, y, angle])

        # update particle positions
        self.particle_positions = self.motion_model.evaluate(self.particle_positions, odometry)

        self.publish_avg_pose()

    def laser_callback(self, msg):
        if len(self.particle_positions) == 0: return

        # evaluate sensor model
        laser_ranges = np.random.choice(np.array(msg.ranges), 100)
        weights = self.sensor_model.evaluate(self.particle_positions, laser_ranges)

        if weights is None:
            return

        # resample particles
        M = len(weights)
        weights /= np.sum(weights) # normalize so they add to 1
        self.particle_samples_indices = np.random.choice(M, size=M, p=weights)

        self.publish_avg_pose()

    def pose_callback(self, msg):
        self.get_logger().info("initial pose")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        angle = 2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        pose = np.array([[x, y , angle]])
        if len(self.particle_positions) != 0:
            self.particle_positions = np.vstack((self.particle_positions, pose)) 
        else:
            self.particle_positions = pose
        # self.get_logger().info("%s" % self.particle_positions)
            
        self.publish_particle_poses()

    def publish_particle_poses(self):
        poses = []
        for i in range(len(self.particle_positions)):
            particle = self.particle_positions[i, :]
            msg = Pose()
            
            msg.position.x = particle[0]
            msg.position.y = particle[1]

            # rotation is around z-axis
            angle = particle[2]
            msg.orientation.x = 0.0
            msg.orientation.y = 0.0
            msg.orientation.z = np.sin(angle / 2)
            msg.orientation.w = np.cos(angle / 2)

            poses.append(msg)

        msg = PoseArray()
        msg.header.frame_id = '/map'

        msg.poses = poses

        self.particles_pub.publish(msg)


        


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()

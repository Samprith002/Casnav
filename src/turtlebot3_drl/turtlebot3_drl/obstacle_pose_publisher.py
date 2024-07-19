#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('obstacle_pose_publisher')

    publishers = []
    for i in range(1, 7):
        pub = node.create_publisher(Odometry, f'/obstacle{i}/pose', 10)
        publishers.append(pub)
    
    node.get_logger().info(f"Created {len(publishers)} publishers")

    # Define initial positions for obstacles
    positions = [
        (2.0, 2.0),  # obstacle1
        (0.0, 2.0),  # obstacle2
        (2.0, 0.0),  # obstacle3
        (-2.0, 0.0), # obstacle4
        (0.0, -2.0), # obstacle5
        (-2.0, -2.0) # obstacle6
    ]

    def publish_poses():
        current_time = node.get_clock().now().to_msg()
        
        for i, (x, y) in enumerate(positions):
            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = "world"
            odom.child_frame_id = f"obstacle{i+1}"
            
            # Position
            odom.pose.pose.position = Point(x=x, y=y, z=0.0)
            
            # Orientation (assuming obstacles don't rotate)
            odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            # Publish the message
            publishers[i].publish(odom)
        
        node.get_logger().debug("Published obstacle poses")

    timer = node.create_timer(0.1, publish_poses)  # 10 Hz

    node.get_logger().info("Obstacle pose publisher started")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
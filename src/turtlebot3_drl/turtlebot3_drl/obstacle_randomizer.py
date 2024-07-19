#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Odometry
import random
from .gst_predictor import load_gst_model, predict_trajectory  # Adjust the import statement based on your project structure
import torch
import os
import math

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('obstacle_randomizer')

    # Set logger level
    node.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

    node.get_logger().info("Obstacle randomizer node started")

    publishers = []
    for i in range(1, 7):
        pub = node.create_publisher(Twist, f'/obstacle{i}/cmd_vel', 10)
        publishers.append(pub)
    node.get_logger().info(f"Created {len(publishers)} publishers")

    # Load the GST model
    model_path = '/home/arjun/Documents/Projects/PILCASNav/CrowdNav_Prediction_AttnGraph/gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj/checkpoint/epoch_100.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    node.get_logger().info("Loading GST model...")
    gst_model = load_gst_model(model_path)
    node.get_logger().info("Model loaded successfully")

    # Subscribe to obstacle poses
    pose_array = PoseArray()

    def pose_callback(msg, obstacle_id):
        nonlocal pose_array
        if len(pose_array.poses) < obstacle_id:
            pose_array.poses.append(msg.pose.pose)
        else:
            pose_array.poses[obstacle_id - 1] = msg.pose.pose
        node.get_logger().debug(f"Received pose for obstacle {obstacle_id}")

    subscriptions = []
    for i in range(1, 7):
        sub = node.create_subscription(
            Odometry,
            f'/obstacle{i}/pose',
            lambda msg, i=i: pose_callback(msg, i),
            10)
        subscriptions.append(sub)
    node.get_logger().info(f"Created {len(subscriptions)} subscriptions")

    def publish_movements():
        nonlocal pose_array
        node.get_logger().info('publish_movements called')
        node.get_logger().info(f'Number of poses received: {len(pose_array.poses)}')

        if len(pose_array.poses) == 6:  # Ensure we have poses for all obstacles
            try:
                # Prepare input for the GST model
                input_data = prepare_input(pose_array)
                node.get_logger().info(f'Input data: {input_data}')

                # Get predictions from the GST model
                predictions = predict_trajectory(gst_model, input_data)
                if predictions is not None:
                    # Use predictions to move obstacles for multiple steps
                    steps = 5  # Define how many steps to predict
                    for i, prediction in enumerate(predictions):
                        twist = Twist()
                        if len(prediction) >= steps:  # Ensure we have enough steps
                            for step in range(steps):
                                twist.linear.x = prediction[step][0]  # Predicted x velocity
                                twist.linear.y = prediction[step][1]  # Predicted y velocity
                                # Calculate angular velocity based on linear velocities if needed
                                twist.angular.z = math.atan2(prediction[step][1], prediction[step][0])
                                publishers[i].publish(twist)
                                node.get_logger().info(f'Predicted movement for obstacle{i+1} at step {step}: linear.x={twist.linear.x:.2f}, linear.y={twist.linear.y:.2f}, angular.z={twist.angular.z:.2f}')
                        else:
                            # Fallback to random movement if prediction is not sufficient
                            twist.linear.x = random.uniform(-0.5, 0.5)
                            twist.linear.y = random.uniform(-0.5, 0.5)
                            twist.angular.z = random.uniform(-1.0, 1.0)
                            publishers[i].publish(twist)
                            node.get_logger().info(f'Fallback random movement for obstacle{i+1}: linear.x={twist.linear.x:.2f}, linear.y={twist.linear.y:.2f}, angular.z={twist.angular.z:.2f}')
                else:
                    node.get_logger().warn("Prediction failed, using random movement")
                    use_random_movement()
            except Exception as e:
                node.get_logger().error(f'Error during prediction: {str(e)}')
                use_random_movement()
        else:
            # Fallback to random movement if we don't have all poses
            node.get_logger().warn("Not all obstacle poses received. Using random movement.")
            use_random_movement()

    def use_random_movement():
        for i, pub in enumerate(publishers):
            twist = Twist()
            twist.linear.x = random.uniform(-0.5, 0.5)
            twist.angular.z = random.uniform(-1.0, 1.0)
            pub.publish(twist)
            node.get_logger().info(f'Random movement for obstacle{i+1}: linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}')

    def prepare_input(pose_array):
        input_data = torch.tensor([[pose.position.x, pose.position.y] for pose in pose_array.poses])
        input_data = input_data.unsqueeze(0).repeat(8, 1, 1)
        return input_data

    timer = node.create_timer(1.0, publish_movements)
    node.get_logger().info("Timer created")

    try:
        node.get_logger().info("Starting to spin...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        node.get_logger().info("Shutting down...")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

"""
Examples using the synchronous client API
Run these examples to see how easy it is to use the client library
"""

from quest_controller_client import QuestControllerClientSync, get_controller_pose


def example_1_basic_query():
    """Example 1: Get a single pose"""
    print("=== Example 1: Basic Query ===\n")

    with QuestControllerClientSync('http://localhost:8000') as client:
        pose = client.get_latest_pose()

        if pose:
            print(f"Timestamp: {pose.timestamp:.3f}")
            print(f"Coordinate System: {pose.coordinate_system}")
            print(f"Latency: {pose.latency:.1f}ms")

            if pose.left:
                x, y, z = pose.left.position
                print(f"\nLeft Controller:")
                print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f})")
                print(f"  Orientation: {pose.left.orientation}")

            if pose.right:
                x, y, z = pose.right.position
                print(f"\nRight Controller:")
                print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f})")
                print(f"  Orientation: {pose.right.orientation}")
        else:
            print("No pose data available")


def example_2_server_status():
    """Example 2: Check server status"""
    print("\n=== Example 2: Server Status ===\n")

    with QuestControllerClientSync('http://localhost:8000') as client:
        if client.is_connected():
            print("✓ Server is reachable")

            status = client.get_status()
            print(f"\nServer Status:")
            print(f"  Uptime: {status.uptime_seconds:.1f}s")
            print(f"  Quest Clients: {status.quest_clients}")
            print(f"  Visualizer Clients: {status.visualizer_clients}")
            print(f"  Frame Rate: {status.current_frame_rate:.1f} Hz")
            print(f"  Total Frames: {status.total_frames_received}")
            print(f"  Has Pose Data: {status.has_pose_data}")
        else:
            print("✗ Server is not reachable")


def example_3_stream_realtime():
    """Example 3: Stream real-time data"""
    print("\n=== Example 3: Real-Time Streaming ===\n")
    print("Streaming for 5 seconds... (Press Ctrl+C to stop)\n")

    frame_count = 0

    def on_pose(pose):
        nonlocal frame_count
        frame_count += 1

        # Clear line and print
        output = f"\rFrame {frame_count}: "

        if pose.left:
            x, y, z = pose.left.position
            output += f"Left=({x:6.3f}, {y:6.3f}, {z:6.3f}) "

        if pose.right:
            x, y, z = pose.right.position
            output += f"Right=({x:6.3f}, {y:6.3f}, {z:6.3f})"

        print(output, end='', flush=True)

    try:
        with QuestControllerClientSync('http://localhost:8000') as client:
            client.poll(on_pose, rate_hz=30, duration=5)
    except KeyboardInterrupt:
        pass

    print(f"\n\nReceived {frame_count} frames")


def example_4_button_states():
    """Example 4: Monitor button states"""
    print("\n=== Example 4: Button States ===\n")
    print("Press buttons on your controllers...\n")

    def on_pose(pose):
        output = []

        if pose.left:
            pressed = [str(i) for i in range(6) if pose.left.is_button_pressed(i)]
            if pressed:
                output.append(f"Left buttons: {', '.join(pressed)}")

        if pose.right:
            pressed = [str(i) for i in range(6) if pose.right.is_button_pressed(i)]
            if pressed:
                output.append(f"Right buttons: {', '.join(pressed)}")

        if output:
            print(' | '.join(output))

    try:
        with QuestControllerClientSync('http://localhost:8000') as client:
            client.poll(on_pose, rate_hz=10, duration=10)
    except KeyboardInterrupt:
        pass


def example_5_iterator():
    """Example 5: Use iterator interface"""
    print("\n=== Example 5: Iterator Interface ===\n")

    count = 0
    max_count = 50

    with QuestControllerClientSync('http://localhost:8000') as client:
        for pose in client.iter_poses(rate_hz=10):
            count += 1

            if pose and pose.left:
                x, y, z = pose.left.position
                print(f"[{count:3d}] Left position: ({x:6.3f}, {y:6.3f}, {z:6.3f})")

            if count >= max_count:
                break


def example_6_quick_access():
    """Example 6: Quick one-shot access"""
    print("\n=== Example 6: Quick Access ===\n")

    # Get latest pose
    pose = get_controller_pose('http://localhost:8000')
    print(f"Latest pose timestamp: {pose.timestamp if pose else 'N/A'}")

    # Get specific controllers
    left = get_controller_pose('http://localhost:8000', controller='left')
    if left:
        print(f"Left position: {left.position}")

    right = get_controller_pose('http://localhost:8000', controller='right')
    if right:
        print(f"Right position: {right.position}")


def example_7_collect_data():
    """Example 7: Collect data for analysis"""
    print("\n=== Example 7: Data Collection ===\n")
    print("Collecting data for 3 seconds...\n")

    positions = {'left': [], 'right': []}

    def collect(pose):
        if pose.left:
            positions['left'].append(pose.left.position)
        if pose.right:
            positions['right'].append(pose.right.position)

    with QuestControllerClientSync('http://localhost:8000') as client:
        client.poll(collect, rate_hz=30, duration=3)

    # Analyze
    print(f"Collected {len(positions['left'])} left positions")
    print(f"Collected {len(positions['right'])} right positions")

    if positions['left']:
        avg_x = sum(p[0] for p in positions['left']) / len(positions['left'])
        avg_y = sum(p[1] for p in positions['left']) / len(positions['left'])
        avg_z = sum(p[2] for p in positions['left']) / len(positions['left'])
        print(f"\nLeft average position: ({avg_x:.3f}, {avg_y:.3f}, {avg_z:.3f})")


def main():
    """Run all examples"""
    examples = [
        example_1_basic_query,
        example_2_server_status,
        example_3_stream_realtime,
        example_4_button_states,
        example_5_iterator,
        example_6_quick_access,
        example_7_collect_data,
    ]

    print("Quest Controller Client - Synchronous Examples")
    print("=" * 50)

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\nError in example {i}: {e}")

        if i < len(examples):
            input("\nPress Enter for next example...")

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == '__main__':
    main()

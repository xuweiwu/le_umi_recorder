"""
Examples using the async client API
For high-performance applications
"""

import asyncio
from quest_controller_client import QuestControllerClient


async def example_1_basic_query():
    """Example 1: Get a single pose (async)"""
    print("=== Example 1: Basic Async Query ===\n")

    async with QuestControllerClient('http://localhost:8000') as client:
        pose = await client.get_latest_pose()

        if pose:
            print(f"Timestamp: {pose.timestamp:.3f}")

            if pose.left:
                print(f"Left: {pose.left.position}")

            if pose.right:
                print(f"Right: {pose.right.position}")
        else:
            print("No pose data available")


async def example_2_server_status():
    """Example 2: Check server status (async)"""
    print("\n=== Example 2: Server Status (Async) ===\n")

    async with QuestControllerClient('http://localhost:8000') as client:
        status = await client.get_status()

        print(f"Server Status:")
        print(f"  Frame Rate: {status.current_frame_rate:.1f} Hz")
        print(f"  Total Frames: {status.total_frames_received}")


async def example_3_stream_realtime():
    """Example 3: Stream real-time data (async)"""
    print("\n=== Example 3: Real-Time Streaming (Async) ===\n")
    print("Streaming for 5 seconds...\n")

    frame_count = 0

    def on_pose(pose):
        nonlocal frame_count
        frame_count += 1
        print(f"Frame {frame_count}: timestamp={pose.timestamp:.3f}")

    client = QuestControllerClient('http://localhost:8000')

    try:
        # Stream for 5 seconds
        stream_task = asyncio.create_task(client.stream(on_pose))
        await asyncio.sleep(5)
        stream_task.cancel()

        try:
            await stream_task
        except asyncio.CancelledError:
            pass

    finally:
        await client.close()

    print(f"\nReceived {frame_count} frames")


async def example_4_async_callback():
    """Example 4: Async callback for async operations"""
    print("\n=== Example 4: Async Callback ===\n")

    frame_count = 0

    async def on_pose_async(pose):
        """Async callback can do async operations"""
        nonlocal frame_count
        frame_count += 1

        # Simulate async operation (e.g., database write)
        await asyncio.sleep(0.001)

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")

    client = QuestControllerClient('http://localhost:8000')

    try:
        stream_task = asyncio.create_task(client.stream_async(on_pose_async))
        await asyncio.sleep(3)
        stream_task.cancel()

        try:
            await stream_task
        except asyncio.CancelledError:
            pass

    finally:
        await client.close()

    print(f"\nTotal frames processed: {frame_count}")


async def example_5_concurrent_operations():
    """Example 5: Concurrent operations"""
    print("\n=== Example 5: Concurrent Operations ===\n")

    async with QuestControllerClient('http://localhost:8000') as client:
        # Run multiple operations concurrently
        status_task = client.get_status()
        pose_task = client.get_latest_pose()
        left_task = client.get_left_controller()
        right_task = client.get_right_controller()

        # Wait for all
        status, pose, left, right = await asyncio.gather(
            status_task,
            pose_task,
            left_task,
            right_task
        )

        print(f"Status: {status.current_frame_rate:.1f} Hz")
        print(f"Pose timestamp: {pose.timestamp if pose else 'N/A'}")
        print(f"Left available: {left is not None}")
        print(f"Right available: {right is not None}")


async def example_6_poll_async():
    """Example 6: Poll at fixed rate (async)"""
    print("\n=== Example 6: Async Polling ===\n")

    frame_count = 0

    def on_pose(pose):
        nonlocal frame_count
        frame_count += 1
        if pose and pose.left:
            print(f"[{frame_count:3d}] Left: {pose.left.position}")

    client = QuestControllerClient('http://localhost:8000')

    try:
        await client.poll(on_pose, rate_hz=5, duration=5)
    finally:
        await client.close()

    print(f"\nPolled {frame_count} times")


async def example_7_connection_callbacks():
    """Example 7: Connection lifecycle callbacks"""
    print("\n=== Example 7: Connection Callbacks ===\n")

    connected = False

    def on_connect():
        nonlocal connected
        connected = True
        print("✓ Connected to server")

    def on_disconnect():
        print("✗ Disconnected from server")

    def on_error(error):
        print(f"! Error: {error}")

    def on_pose(pose):
        if connected:
            print(f"  Pose: {pose.timestamp:.3f}")

    client = QuestControllerClient('http://localhost:8000')

    try:
        stream_task = asyncio.create_task(
            client.stream(
                on_pose,
                on_connect=on_connect,
                on_disconnect=on_disconnect,
                on_error=on_error
            )
        )

        await asyncio.sleep(3)
        stream_task.cancel()

        try:
            await stream_task
        except asyncio.CancelledError:
            pass

    finally:
        await client.close()


async def main():
    """Run all async examples"""
    examples = [
        example_1_basic_query,
        example_2_server_status,
        example_3_stream_realtime,
        example_4_async_callback,
        example_5_concurrent_operations,
        example_6_poll_async,
        example_7_connection_callbacks,
    ]

    print("Quest Controller Client - Async Examples")
    print("=" * 50)

    for i, example in enumerate(examples, 1):
        try:
            await example()
        except Exception as e:
            print(f"\nError in example {i}: {e}")

        if i < len(examples):
            await asyncio.sleep(1)

    print("\n" + "=" * 50)
    print("All async examples completed!")


if __name__ == '__main__':
    asyncio.run(main())

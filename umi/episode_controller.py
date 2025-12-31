"""
Episode Controller for UMI Recording
State machine for managing recording episodes with keyboard/controller input
"""

import sys
import time
import threading
from enum import Enum
from typing import Callable, Optional

# Import PoseData type for type hints
try:
    sys.path.insert(0, str(__file__).replace('/umi/episode_controller.py', '/backend'))
    from api import PoseData
except ImportError:
    PoseData = None  # Type hint only


class EpisodeState(Enum):
    """States for episode recording."""
    IDLE = "idle"           # Not recording, waiting for start
    RECORDING = "recording" # Actively recording frames
    SAVING = "saving"       # Saving episode to disk


class EpisodeController:
    """
    State machine for episode recording control.

    Supports two control modes:
    - keyboard: 's' to start, 'e' to end, 'c' to cancel, 'q' to quit
    - controller: Trigger button hold >0.5s to toggle recording

    Usage:
        controller = EpisodeController(control_mode='keyboard')
        controller.start()

        # In main loop:
        if controller.should_start():
            # Start recording
        if controller.should_stop():
            # Stop recording
        if controller.should_quit():
            # Exit application

        controller.stop()
    """

    def __init__(
        self,
        control_mode: str = 'keyboard',
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize episode controller.

        Args:
            control_mode: 'keyboard' or 'controller'
            on_start: Callback when episode starts
            on_stop: Callback when episode ends (save)
            on_cancel: Callback when episode is cancelled (discard)
        """
        self.control_mode = control_mode
        self.state = EpisodeState.IDLE
        self.episode_count = 0

        # Callbacks
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_cancel = on_cancel

        # Keyboard state
        self._keyboard_listener = None
        self._pending_action: Optional[str] = None
        self._quit_requested = False
        self._lock = threading.Lock()

        # Controller trigger state
        self._trigger_held = False
        self._trigger_start_time = 0.0
        self._trigger_action_fired = False

    def start(self):
        """Start the episode controller (keyboard listener if needed)."""
        if self.control_mode == 'keyboard':
            self._start_keyboard_listener()
        print(f"Episode controller started in {self.control_mode} mode")
        self._print_controls()

    def stop(self):
        """Stop the episode controller."""
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def _print_controls(self):
        """Print control instructions."""
        if self.control_mode == 'keyboard':
            print("\nControls:")
            print("  [s] Start recording episode")
            print("  [e] End episode (save)")
            print("  [c] Cancel episode (discard)")
            print("  [q] Quit application")
            print()
        else:
            print("\nControls:")
            print("  Hold trigger >0.5s to toggle recording")
            print("  Press Ctrl+C to quit")
            print()

    def _start_keyboard_listener(self):
        """Start listening for keyboard input."""
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    char = key.char.lower() if hasattr(key, 'char') and key.char else None
                    if char:
                        with self._lock:
                            if char == 's':
                                self._pending_action = 'start'
                            elif char == 'e':
                                self._pending_action = 'stop'
                            elif char == 'c':
                                self._pending_action = 'cancel'
                            elif char == 'q':
                                self._quit_requested = True
                except Exception:
                    pass

            self._keyboard_listener = keyboard.Listener(on_press=on_press)
            self._keyboard_listener.start()

        except ImportError:
            print("Warning: pynput not installed. Keyboard control disabled.")
            print("Install with: pip install pynput")
            # Fall back to simple input-based control
            self._start_simple_input_thread()

    def _start_simple_input_thread(self):
        """Fallback: use stdin for input (blocking)."""
        def input_thread():
            while not self._quit_requested:
                try:
                    line = input()
                    with self._lock:
                        if line.lower() == 's':
                            self._pending_action = 'start'
                        elif line.lower() == 'e':
                            self._pending_action = 'stop'
                        elif line.lower() == 'c':
                            self._pending_action = 'cancel'
                        elif line.lower() == 'q':
                            self._quit_requested = True
                except EOFError:
                    break

        thread = threading.Thread(target=input_thread, daemon=True)
        thread.start()

    def check_controller_input(self, pose: 'PoseData') -> Optional[str]:
        """
        Check controller button for episode control.

        Args:
            pose: Current pose data with button states

        Returns:
            'toggle' if trigger held >0.5s, None otherwise
        """
        if pose is None:
            return None

        # Check left controller trigger (button index 0)
        trigger_value = 0.0
        if pose.left and pose.left.buttons:
            btn = pose.left.buttons.get('0', pose.left.buttons.get(0, {}))
            trigger_value = btn.get('value', 0.0) if isinstance(btn, dict) else 0.0

        # Also check right controller
        if pose.right and pose.right.buttons:
            btn = pose.right.buttons.get('0', pose.right.buttons.get(0, {}))
            right_value = btn.get('value', 0.0) if isinstance(btn, dict) else 0.0
            trigger_value = max(trigger_value, right_value)

        # Detect trigger hold
        if trigger_value > 0.8:
            if not self._trigger_held:
                self._trigger_held = True
                self._trigger_start_time = time.time()
                self._trigger_action_fired = False
            elif not self._trigger_action_fired and (time.time() - self._trigger_start_time) > 0.5:
                self._trigger_action_fired = True
                return 'toggle'
        else:
            self._trigger_held = False
            self._trigger_action_fired = False

        return None

    def process_input(self, pose: Optional['PoseData'] = None) -> Optional[str]:
        """
        Process input and return any pending action.

        Args:
            pose: Current pose data (needed for controller mode)

        Returns:
            'start', 'stop', 'cancel', or None
        """
        # Check keyboard input
        with self._lock:
            if self._pending_action:
                action = self._pending_action
                self._pending_action = None
                return action

        # Check controller input
        if self.control_mode == 'controller' and pose is not None:
            toggle = self.check_controller_input(pose)
            if toggle == 'toggle':
                # Toggle between IDLE and RECORDING
                if self.state == EpisodeState.IDLE:
                    return 'start'
                elif self.state == EpisodeState.RECORDING:
                    return 'stop'

        return None

    def transition(self, action: str) -> bool:
        """
        Attempt state transition based on action.

        Args:
            action: 'start', 'stop', or 'cancel'

        Returns:
            True if transition was successful
        """
        if action == 'start':
            if self.state == EpisodeState.IDLE:
                self.state = EpisodeState.RECORDING
                self.episode_count += 1
                print(f"\n>>> Recording episode {self.episode_count}...")
                if self.on_start:
                    self.on_start()
                return True
            else:
                print("Cannot start: already recording")
                return False

        elif action == 'stop':
            if self.state == EpisodeState.RECORDING:
                self.state = EpisodeState.SAVING
                print(f">>> Saving episode {self.episode_count}...")
                if self.on_stop:
                    self.on_stop()
                self.state = EpisodeState.IDLE
                return True
            else:
                print("Cannot stop: not recording")
                return False

        elif action == 'cancel':
            if self.state == EpisodeState.RECORDING:
                print(f">>> Cancelling episode {self.episode_count}...")
                if self.on_cancel:
                    self.on_cancel()
                self.episode_count -= 1  # Don't count cancelled episodes
                self.state = EpisodeState.IDLE
                return True
            else:
                print("Cannot cancel: not recording")
                return False

        return False

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.state == EpisodeState.RECORDING

    @property
    def should_quit(self) -> bool:
        """Check if quit was requested."""
        with self._lock:
            return self._quit_requested

    def request_quit(self):
        """Request application quit."""
        with self._lock:
            self._quit_requested = True


def main():
    """Test episode controller."""
    print("Testing EpisodeController in keyboard mode")
    print("Press s/e/c/q to test state transitions")

    controller = EpisodeController(control_mode='keyboard')
    controller.start()

    try:
        while not controller.should_quit:
            action = controller.process_input()
            if action:
                controller.transition(action)
                print(f"State: {controller.state.value}, Episodes: {controller.episode_count}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        print("\nDone!")


if __name__ == '__main__':
    main()

"""Mock-mode tests for ros2_bridge.ROS2TeamAPI (no ROS needed).

Runs a skill program that swaps the banana and the pear, then checks
the recorded ROS goals.
"""

from ros2_bridge import ROS2TeamAPI


def swap_banana_pear(robot):
    """Skill program: swap the banana and the pear via station C parking."""
    banana_at = robot.recall("banana")
    pear_at = robot.recall("pear")
    assert banana_at and pear_at, "both fruits must be on the map"

    assert robot.go_to(banana_at)
    assert robot.sees("banana")
    assert robot.pick("banana")
    assert robot.is_holding()

    assert robot.go_to("station C")  # park the banana elsewhere
    assert robot.put_aside()
    assert not robot.is_holding()

    assert robot.go_to(pear_at)
    assert robot.pick("pear")
    assert robot.go_to(banana_at)  # pear takes the banana's old table
    assert robot.put_aside()

    assert robot.go_to("station C")
    assert robot.pick("banana")
    assert robot.go_to(pear_at)  # banana takes the pear's old table
    assert robot.put_aside()
    return True


def _actions(robot, name):
    return [g for g in robot.goals if g["action"] == name]


def test_swap_banana_pear_mock():
    robot = ROS2TeamAPI(mock=True)
    assert set(robot.places()) == {"station A", "station B",
                                   "station C", "packing station"}

    assert swap_banana_pear(robot) is True

    # The fruits traded tables.
    assert robot.recall("banana") == "station B"
    assert robot.recall("pear") == "station A"
    assert not robot.is_holding()

    # Nav2 saw one goal per drive, in driving order.
    nav = _actions(robot, "NavigateToPose")
    assert [g["place"] for g in nav] == ["station A", "station C",
                                         "station B", "station A",
                                         "station C", "station B"]
    assert all(set(g["pose"]) == {"x", "y", "yaw"} for g in nav)

    # 3 picks (open + close each) + 3 puts (open each), each with arm moves.
    grips = _actions(robot, "GripperCommand")
    assert [g["command"] for g in grips].count("close") == 3
    assert [g["command"] for g in grips].count("open") == 6
    moves = _actions(robot, "MoveGroup")
    assert len(moves) >= 6  # at least one arm move per pick/put
    assert all(len(g["target"]) == 3 for g in moves)


def test_method_parity_with_team_api():
    """Same method names and return types as TeamAPI's ground-robot set."""
    from team_skills import TeamAPI  # signature reference only (not run)
    for name in ("places", "go_to", "pick", "place_on",
                 "put_aside", "sees", "is_holding", "recall"):
        assert hasattr(TeamAPI, name), name
        assert callable(getattr(ROS2TeamAPI, name)), name

    robot = ROS2TeamAPI(mock=True)
    assert isinstance(robot.places(), list)
    assert robot.go_to("station A") is True
    assert robot.go_to("nowhere") is False
    assert robot.pick("banana") is True
    assert robot.is_holding() is True
    assert robot.put_aside("left") is True
    assert robot.sees("tray") is True
    assert robot.recall("banana") == "station A"
    assert robot.recall("missing fruit") is None


def test_mock_refusals_match_team_api():
    robot = ROS2TeamAPI(mock=True)
    assert robot.go_to("station A") is True
    assert robot.pick("") is False  # invalid name, like TeamAPI
    assert robot.pick("unknown thing") is False
    assert robot.put_aside() is False  # not holding anything
    assert robot.place_on("tray") is False
    assert robot.put_aside("up") is False  # bad side, like TeamAPI


# ---------- extra checks added in review (no ROS needed) ----------

import math
import types

import ros2_bridge
from ros2_bridge import hand_down_quaternion, spot_in_front


def test_places_match_the_simulator():
    """places.yaml = sim park poses: table center - 0.85 m along the table's direction."""
    from mission_tasks import STATIONS
    robot = ROS2TeamAPI(mock=True)
    stations = dict(STATIONS)
    stations["packing station"] = ((0.0, -1.6), -math.pi / 2)
    for name, ((cx, cy), yaw) in stations.items():
        p = robot._places[name]
        assert abs(p["x"] - (cx - 0.85 * math.cos(yaw))) < 1e-3, name
        assert abs(p["y"] - (cy - 0.85 * math.sin(yaw))) < 1e-3, name
        assert abs(math.remainder(p["yaw"] - yaw, 2 * math.pi)) < 1e-3, name


def test_put_aside_uses_the_robot_heading():
    """At station B the robot faces +y, so 'left' is -x and the table is at larger y."""
    robot = ROS2TeamAPI(mock=True)
    robot.go_to("station B")
    assert robot.pick("pear")
    assert robot.put_aside("left")
    moves = [g["target"] for g in robot.goals if g["action"] == "MoveGroup"]
    x, y, z = moves[-2]                       # the put-down point (last move is the hand going up)
    assert abs(x - (-0.15)) < 1e-4 and abs(y - (0.75 + ros2_bridge.REACH)) < 1e-4
    assert abs(z - ros2_bridge.TABLE_Z) < 1e-6


def test_fake_detection_is_on_the_table_in_front():
    robot = ROS2TeamAPI(mock=True)
    assert robot._fake_detect("banana") is None          # not parked anywhere yet
    robot.go_to("station C")                             # faces -x
    robot._map["soup can"] = "station C"
    x, y, _ = robot._fake_detect("soup can")
    assert abs(x - (-0.75 - ros2_bridge.REACH)) < 1e-4 and abs(y) < 1e-4


def test_hand_down_quaternion_matches_pybullet():
    import pybullet as p
    for yaw in (0.0, 0.7, math.pi / 2, -2.0, math.pi):
        a = hand_down_quaternion(yaw)
        b = p.getQuaternionFromEuler([math.pi, 0, yaw])
        same = all(abs(u - v) < 1e-6 for u, v in zip(a, b))
        flipped = all(abs(u + v) < 1e-6 for u, v in zip(a, b))   # q and -q are the same rotation
        assert same or flipped, (yaw, a, b)


def test_spot_in_front_left_is_plus_lateral():
    s = spot_in_front({"x": 0.0, "y": 0.0, "yaw": 0.0}, side_offset=0.15, reach=0.6, z=0.45)
    assert [round(v, 6) for v in s] == [0.6, 0.15, 0.45]


def _img_msg(arr, encoding, pad=0):
    import numpy as np
    h, w = arr.shape[:2]
    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(h, -1)
    raw = np.hstack([raw, np.zeros((h, pad), np.uint8)])          # row padding
    return types.SimpleNamespace(height=h, width=w, step=raw.shape[1], encoding=encoding,
                                 is_bigendian=False, data=raw.tobytes(),
                                 header=types.SimpleNamespace(frame_id="camera_color_optical_frame"))


def test_depth_and_color_decoding():
    import numpy as np
    robot = ROS2TeamAPI(mock=True)
    mm = np.array([[500, 1000, 0], [1500, 2000, 2500]], np.uint16)
    robot._on_depth(_img_msg(mm, "16UC1", pad=2))
    assert robot._depth.dtype == np.float32 and abs(robot._depth[1, 2] - 2.5) < 1e-6
    m = np.array([[0.5, 1.25], [2.0, 3.0]], np.float32)
    robot._on_depth(_img_msg(m, "32FC1"))
    assert abs(robot._depth[0, 1] - 1.25) < 1e-6
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[..., 0] = 200                                     # blue in BGR
    robot._on_rgb(_img_msg(bgr, "bgr8", pad=3))
    assert robot._rgb.shape == (2, 3, 3) and robot._rgb[0, 0, 2] == 200 and robot._rgb[0, 0, 0] == 0


def test_action_call_waits_for_the_result(monkeypatch):
    """_run_action: send goal -> accepted -> result -> SUCCEEDED (with a fake rclpy + client)."""
    class Fut:
        def __init__(self, value): self.value = value
        def result(self): return self.value
        def done(self): return True

    class Handle:
        accepted = True
        def get_result_async(self): return Fut(types.SimpleNamespace(status=4))
        def cancel_goal_async(self): pass

    class Client:
        sent = []
        def wait_for_server(self, timeout_sec): return True
        def send_goal_async(self, goal): Client.sent.append(goal); return Fut(Handle())

    fake_rclpy = types.SimpleNamespace(spin_until_future_complete=lambda node, fut, timeout_sec=None: None)
    monkeypatch.setattr(ros2_bridge, "rclpy", fake_rclpy)
    robot = ROS2TeamAPI(mock=True)
    robot._node = types.SimpleNamespace(get_logger=lambda: types.SimpleNamespace(error=print, warn=print))
    robot._msg = {"GoalStatus": types.SimpleNamespace(STATUS_SUCCEEDED=4)}
    assert robot._run_action(Client(), "goal", timeout=1.0) is True
    Handle.accepted = False
    assert robot._run_action(Client(), "goal", timeout=1.0) is False


def test_places_without_pyyaml(monkeypatch):
    monkeypatch.setattr(ros2_bridge, "yaml", None)
    robot = ROS2TeamAPI(mock=True)
    assert robot._places["station B"] == {"x": 0.0, "y": 0.75, "yaw": 1.5708}
    assert len(robot.places()) == 4

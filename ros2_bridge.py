"""ROS 2 bridge with the same ground-robot API as TeamAPI (team_skills.py).

First version written with Muse Code during the hackathon, then reviewed and fixed
(real rclpy action calls, MoveIt pose goals, TF + camera intrinsics, depth encodings,
poses that match the simulator). MOCK mode is tested (test_ros2_bridge.py).
ROS mode has NOT been run on a real robot yet.

Why: the program Muse Spark writes in the simulator calls robot.go_to(), robot.pick(), ...
This class has the same methods, so the same program can drive a real ROS 2 robot.

Covered methods (same names + return values as TeamAPI):
    places() -> list[str]
    go_to(place) -> bool
    pick(item) -> bool
    place_on(thing) -> bool
    put_aside(side=None) -> bool
    sees(item) -> bool
    is_holding() -> bool
    recall(item) -> place name or None

ROS mapping:
    go_to       -> Nav2 NavigateToPose action (poses come from places.yaml)
    pick / put_aside / place_on -> MoveIt 2 MoveGroup action (pose goal for the hand)
    grasp point -> SAM 3.1 box (perception.sam_find) on the colour image
                   + the depth pixel + camera intrinsics + TF to the map frame
    gripper     -> control_msgs GripperCommand action

Modes:
    ROS mode  - used only when rclpy and the message packages are installed.
    MOCK mode - otherwise (or when mock=True): no ROS needed. Every ROS goal is
                recorded in self.goals, a fake detector stands in for SAM 3.1,
                and every action reports success.
"""

import math
import os
import re

# rclpy exists only on a ROS 2 machine. Importing it here decides the default mode.
try:
    import rclpy
except ImportError:
    rclpy = None

try:
    import yaml
except ImportError:
    yaml = None

# ---------- settings for a real robot (change these to match yours) ----------
MAP_FRAME = "map"
NAV_ACTION = "navigate_to_pose"
MOVE_ACTION = "move_action"
GRIP_ACTION = "panda_hand_controller/gripper_cmd"
ARM_GROUP = "panda_arm"
HAND_LINK = "panda_hand_tcp"
COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"   # depth aligned to the colour image
INFO_TOPIC = "/camera/color/camera_info"
NAV_TIMEOUT = 120.0     # seconds for one drive
ARM_TIMEOUT = 30.0      # seconds for one arm move or gripper command

# Table geometry, the same as the simulator (mobile_env.add_station):
# the robot parks 0.85 m from the table center, items sit about 0.62 m in front of it.
REACH = 0.62            # metres in front of the parked robot
TABLE_Z = 0.45          # table top height (m)
ABOVE = 0.15            # approach height above the grasp point (m)

# Lateral offsets (m) for put_aside, as seen by the robot.
# + is the robot's left, matching table_side() in team_skills.py.
SIDE_OFFSET = {"left": 0.15, "middle": 0.0, "right": -0.15}

# Park poses from the simulator: table center - 0.85 m along the table's direction.
DEFAULT_PLACES = {"station A": {"x": 0.75, "y": 0.0, "yaw": 0.0},
                  "station B": {"x": 0.0, "y": 0.75, "yaw": 1.5708},
                  "station C": {"x": -0.75, "y": 0.0, "yaw": 3.1416},
                  "packing station": {"x": 0.0, "y": -0.75, "yaw": -1.5708}}


def hand_down_quaternion(yaw):
    """Gripper pointing down, turned to `yaw` (same as mobile_env.down()). Returns (x, y, z, w)."""
    return (math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0)


def spot_in_front(pose, side_offset=0.0, reach=REACH, z=TABLE_Z):
    """A point on the table in front of a parked robot. + side_offset = the robot's left."""
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    return [pose["x"] + reach * c - side_offset * s,
            pose["y"] + reach * s + side_offset * c,
            z]


class ROS2TeamAPI:
    def __init__(self, places_file="places.yaml", initial_map=None, mock=None):
        # mock=None -> auto: MOCK unless rclpy is installed.
        self.mock = (rclpy is None) if mock is None else bool(mock)
        self.goals = []          # every ROS goal sent (or faked), in order
        self._places = self._load_places(places_file)
        # Where each item was last seen: {"banana": "station A", ...}.
        # Defaults match the mock swap test; pass {} for "nothing known".
        if initial_map is None:
            initial_map = {"banana": "station A", "pear": "station B"}
        self._map = dict(initial_map)
        self._holding = None     # name of the held item, or None
        self._at = None          # place the robot is parked at, or None
        # Latest camera data (ROS mode only).
        self._rgb = None
        self._depth = None       # metres, float32
        self._K = None           # (fx, fy, cx, cy)
        self._cam_frame = None
        self._node = None
        if not self.mock:
            self._init_ros()

    # ---------- setup ----------

    def _load_places(self, places_file):
        """Read {place: {x, y, yaw}} from places.yaml."""
        # Also look next to this file, so tests pass from any folder.
        candidates = [places_file, os.path.join(os.path.dirname(os.path.abspath(__file__)), places_file)]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return {k: dict(v) for k, v in DEFAULT_PLACES.items()}
        if yaml is not None:
            with open(path) as f:
                raw = yaml.safe_load(f)["places"]
            return {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in raw.items()}
        # Tiny fallback parser for our one-place-per-line file (no pyyaml).
        out = {}
        pat = re.compile(r"(.+?):\s*\{([^}]*)\}")
        with open(path) as f:
            for line in f:
                if line.strip().startswith("#"):
                    continue
                m = pat.search(line)
                if m:
                    name = m.group(1).strip().strip('"')
                    vals = dict(kv.split(":") for kv in m.group(2).split(","))
                    out[name] = {k.strip(): float(v) for k, v in vals.items()}
        return out

    def _init_ros(self):
        """Create the node, action clients, TF listener and camera subscriptions."""
        # All ROS imports live here so MOCK mode never needs ROS installed.
        try:
            from action_msgs.msg import GoalStatus
            from control_msgs.action import GripperCommand
            from geometry_msgs.msg import Pose
            from moveit_msgs.action import MoveGroup
            from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
            from rclpy.node import Node
            from sensor_msgs.msg import CameraInfo, Image
            from shape_msgs.msg import SolidPrimitive
            import tf2_ros
        except ImportError:
            # rclpy without the message packages: stay usable in MOCK mode.
            self.mock = True
            return
        self._msg = dict(GoalStatus=GoalStatus, GripperCommand=GripperCommand, MoveGroup=MoveGroup, Pose=Pose,
                         Constraints=Constraints, OrientationConstraint=OrientationConstraint,
                         PositionConstraint=PositionConstraint, NavigateToPose=NavigateToPose,
                         SolidPrimitive=SolidPrimitive)
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("hey_robot_team_bridge")
        self._nav_client = ActionClient(self._node, NavigateToPose, NAV_ACTION)
        self._move_client = ActionClient(self._node, MoveGroup, MOVE_ACTION)
        self._grip_client = ActionClient(self._node, GripperCommand, GRIP_ACTION)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        self._node.create_subscription(Image, COLOR_TOPIC, self._on_rgb, 10)
        self._node.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 10)
        self._node.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, 10)

    # ---------- camera callbacks (no cv_bridge needed) ----------

    @staticmethod
    def _image_array(msg, dtype, channels):
        import numpy as np
        row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        width_bytes = msg.width * channels * np.dtype(dtype).itemsize
        arr = row[:, :width_bytes].copy().view(dtype)       # drop row padding, if any
        if msg.is_bigendian != (np.dtype(dtype).byteorder == ">"):
            arr = arr.byteswap() if np.dtype(dtype).itemsize > 1 else arr
        return arr.reshape(msg.height, msg.width, channels) if channels > 1 else arr.reshape(msg.height, msg.width)

    def _on_rgb(self, msg):
        import numpy as np
        if msg.encoding in ("rgb8", "bgr8"):
            img = self._image_array(msg, np.uint8, 3)
            self._rgb = img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img
        elif msg.encoding in ("rgba8", "bgra8"):
            img = self._image_array(msg, np.uint8, 4)[:, :, :3]
            self._rgb = img[:, :, ::-1].copy() if msg.encoding == "bgra8" else img.copy()

    def _on_depth(self, msg):
        import numpy as np
        if msg.encoding in ("16UC1", "mono16"):          # millimetres (RealSense, many others)
            self._depth = self._image_array(msg, np.uint16, 1).astype(np.float32) / 1000.0
        elif msg.encoding == "32FC1":                     # metres
            self._depth = self._image_array(msg, np.float32, 1)
        self._cam_frame = msg.header.frame_id

    def _on_info(self, msg):
        self._K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])   # fx, fy, cx, cy

    def _record(self, action, **detail):
        goal = {"action": action}
        goal.update(detail)
        self.goals.append(goal)
        return goal

    # ---------- TeamAPI-compatible methods ----------

    def places(self):
        return list(self._places)

    def go_to(self, place):
        place = str(place)
        if place not in self._places:
            return False
        pose = self._places[place]
        self._record("NavigateToPose", place=place, pose=dict(pose))
        ok = True if self.mock else self._send_nav(pose)
        if ok:
            self._at = place
        return ok

    def pick(self, item):
        item = str(item).strip()
        # Same refusals as TeamAPI.pick.
        if not item or "unknown" in item.lower():
            return False
        if self._holding is not None:        # one item at a time
            return False
        grasp = self._detect(item)           # SAM 3.1 box center + depth -> map frame
        if grasp is None:                    # could not see it here
            return False
        x, y, z = grasp
        ok = (self._gripper("open")
              and self._move([x, y, z + ABOVE], item)     # arrive from above
              and self._move([x, y, z], item)             # go to the grasp point
              and self._gripper("close"))
        if not ok:
            return False
        self._move([x, y, z + ABOVE], item)               # lift clear
        self._holding = item
        self._map.pop(item, None)            # no longer on its old table
        return True

    def place_on(self, thing):
        if self._holding is None:
            return False
        target = self._detect(str(thing))    # SAM 3.1 finds the tray / table
        if target is None:
            return False
        x, y, z = target
        if not self._move([x, y, z + ABOVE], self._holding):
            return False
        self._gripper("open")                # let go above the thing
        self._move([x, y, z + ABOVE + 0.10], self._holding)   # retreat straight up
        self._map[self._holding] = self._at  # now rests at this place
        self._holding = None
        return True

    def put_aside(self, side=None):
        side = str(side).lower().strip() if side else None
        if side not in (None, "left", "middle", "right"):
            return False                     # same refusal as TeamAPI
        if self._holding is None or self._at is None:
            return False
        # Free spot on the table in front of the parked robot (uses its heading).
        spot = spot_in_front(self._places[self._at], SIDE_OFFSET[side or "middle"])
        if not self._move([spot[0], spot[1], spot[2] + ABOVE], self._holding):
            return False
        self._move(spot, self._holding)
        self._gripper("open")
        self._move([spot[0], spot[1], spot[2] + ABOVE], self._holding)   # hand up again
        self._map[self._holding] = self._at
        self._holding = None
        return True

    def sees(self, item):
        item = str(item)
        if self.mock:
            # Fake detector: an item is seen on its mapped table; the held item is in the
            # hand (not on the table); furniture like "tray" is always assumed visible.
            if item == self._holding:
                return False
            if item in self._map:
                return self._map[item] == self._at
            return True
        return self._detect(item) is not None

    def is_holding(self):
        return self._holding is not None

    def recall(self, item):
        return self._map.get(str(item))

    # ---------- perception: SAM 3.1 box + depth -> grasp point in the map frame ----------

    def _detect(self, phrase):
        if self.mock:
            return self._fake_detect(phrase)
        return self._ros_detect(phrase)

    def _fake_detect(self, phrase):
        """Stand-in for SAM: the middle of the table in front of the parked robot."""
        if self._at is None or not self.sees(phrase):
            return None
        return spot_in_front(self._places[self._at])

    def _ros_detect(self, phrase):
        """Best SAM 3.1 box center, deprojected with its depth pixel, moved into the map frame."""
        import numpy as np
        from perception import sam_find      # the same Meta Model API call as the simulator
        for _ in range(50):                  # wait up to 5 s for colour, depth and intrinsics
            if self._rgb is not None and self._depth is not None and self._K is not None:
                break
            rclpy.spin_once(self._node, timeout_sec=0.1)
        else:
            return None
        boxes = sam_find(self._rgb, phrase)
        if not boxes:
            return None
        best = max(boxes, key=lambda b: b.get("conf") or 0)
        cx, cy = best["center"]
        # median depth of a small patch (one pixel is often 0 on real depth cameras)
        h, w = self._depth.shape
        u, v = int(round(cx)), int(round(cy))
        patch = self._depth[max(0, v - 3):min(h, v + 4), max(0, u - 3):min(w, u + 4)]
        good = patch[np.isfinite(patch) & (patch > 0.1)]
        if good.size == 0:
            return None
        depth = float(np.median(good))
        fx, fy, ux, uy = self._K
        cam = ((cx - ux) * depth / fx, (cy - uy) * depth / fy, depth)   # optical frame
        xyz = self._camera_to_map(cam)
        self._record("Perception", phrase=phrase, box_center=[cx, cy], depth_m=round(depth, 3),
                     map_xyz=None if xyz is None else [round(c, 3) for c in xyz])
        return xyz

    def _camera_to_map(self, cam):
        """Camera optical-frame point -> map frame, through TF."""
        from geometry_msgs.msg import PointStamped
        from rclpy.duration import Duration
        from rclpy.time import Time
        import tf2_geometry_msgs  # noqa: F401  (registers PointStamped for tf2)
        pt = PointStamped()
        pt.header.frame_id = self._cam_frame
        pt.point.x, pt.point.y, pt.point.z = cam
        try:
            tf = self._tf_buffer.lookup_transform(MAP_FRAME, self._cam_frame, Time(),
                                                  timeout=Duration(seconds=1.0))
            out = tf2_geometry_msgs.do_transform_point(pt, tf)
        except Exception as e:               # no TF yet: do not guess a grasp point
            self._node.get_logger().warn(f"TF {self._cam_frame} -> {MAP_FRAME} failed: {e}")
            return None
        return [out.point.x, out.point.y, out.point.z]

    # ---------- ROS goals (recorded always, sent only in ROS mode) ----------

    def _run_action(self, client, goal, timeout):
        """Send one action goal and wait for its result. True only if it succeeded."""
        if not client.wait_for_server(timeout_sec=5.0):
            self._node.get_logger().error("action server not available")
            return False
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send, timeout_sec=10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return False
        done = handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, done, timeout_sec=timeout)
        if not done.done():                  # took too long: cancel and report failure
            handle.cancel_goal_async()
            return False
        return done.result().status == self._msg["GoalStatus"].STATUS_SUCCEEDED

    def _send_nav(self, pose):
        """Drive to a named pose with Nav2."""
        goal = self._msg["NavigateToPose"].Goal()
        goal.pose.header.frame_id = MAP_FRAME
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = pose["x"]
        goal.pose.pose.position.y = pose["y"]
        goal.pose.pose.orientation.z = math.sin(pose["yaw"] / 2)    # flat yaw -> quaternion
        goal.pose.pose.orientation.w = math.cos(pose["yaw"] / 2)
        return self._run_action(self._nav_client, goal, NAV_TIMEOUT)

    def _move(self, xyz, item):
        """Move the hand to xyz (map frame), pointing down, with a MoveIt 2 pose goal."""
        yaw = self._places[self._at]["yaw"] if self._at in self._places else 0.0
        self._record("MoveGroup", target=list(xyz), item=item, frame=MAP_FRAME)
        if self.mock:
            return True
        m = self._msg
        goal = m["MoveGroup"].Goal()
        req = goal.request
        req.group_name = ARM_GROUP
        req.planner_id = "RRTConnectkConfigDefault"
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3
        # position: a 1 cm sphere around the target
        pos = m["PositionConstraint"]()
        pos.header.frame_id = MAP_FRAME
        pos.link_name = HAND_LINK
        ball = m["SolidPrimitive"]()
        ball.type = m["SolidPrimitive"].SPHERE
        ball.dimensions = [0.01]
        pos.constraint_region.primitives.append(ball)
        center = m["Pose"]()
        center.position.x, center.position.y, center.position.z = xyz
        center.orientation.w = 1.0
        pos.constraint_region.primitive_poses.append(center)
        pos.weight = 1.0
        # orientation: hand pointing down, turned with the robot
        ori = m["OrientationConstraint"]()
        ori.header.frame_id = MAP_FRAME
        ori.link_name = HAND_LINK
        qx, qy, qz, qw = hand_down_quaternion(yaw)
        ori.orientation.x, ori.orientation.y, ori.orientation.z, ori.orientation.w = qx, qy, qz, qw
        ori.absolute_x_axis_tolerance = 0.05
        ori.absolute_y_axis_tolerance = 0.05
        ori.absolute_z_axis_tolerance = 0.1
        ori.weight = 1.0
        c = m["Constraints"]()
        c.position_constraints.append(pos)
        c.orientation_constraints.append(ori)
        req.goal_constraints.append(c)
        goal.planning_options.plan_only = False      # plan and execute
        return self._run_action(self._move_client, goal, ARM_TIMEOUT)

    def _gripper(self, command):
        """Open or close the gripper with the GripperCommand action."""
        self._record("GripperCommand", command=command, holding=self._holding)
        if self.mock:
            return True
        goal = self._msg["GripperCommand"].Goal()
        goal.command.position = 0.04 if command == "open" else 0.0
        goal.command.max_effort = 20.0
        ok = self._run_action(self._grip_client, goal, ARM_TIMEOUT)
        # closing on an object often stops early ("stalled"); that is a good grasp, not a failure
        return ok or command == "close"

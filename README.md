# Hey Robot — a voice-controlled robot team that asks Meta's models before every move

**Meta Global AI Developer Hackathon · Indie track · Problem statement: Push the Frontier (robotics) · Vanapalli Raghava Ashok**

You speak an order. A drone and a wheeled robot with an arm work out the plan, look before they act,
check their own work, and tell you out loud what they did. Every AI step is a Meta model.

> 🎥 Demo video: _add link_  ·  Everything runs in a PyBullet warehouse simulation on a laptop.

```
"Put the fruit in the tray and the tools on station A."
   → Muse Voice Transcribe hears it
   → Muse Spark picks the skill and writes the robot program
   → Muse Image imagines the finished scene
   → the drone scouts the tables with SAM 3.1; Spark confirms every product
   → the robot picks each item only after SAM 3.1 + Spark agree on what and where it is
   → an independent checker and Spark (real camera vs imagined goal) verify the result
   → the robot says "Done. The job is finished and checked."
```

---

## Built during the hackathon (Oct 27 – Nov 4, 2026)

All Hey Robot code in this repository was written during the hackathon building period: the voice
command center, the Muse Spark planner and lesson memory, the Muse Wonder diagnosis loop, the imagination
checks with Muse Image, SAM 3.1 perception and Spark confirmation, the drone (autopilot, SAM-guided landing,
RL landing reflex, vacuum gripper, hand-off on the move), the mobile manipulator skills, the control room,
the spoken replies, the ROS 2 bridge, all 9 skills and all experiments. The only outside parts are the 3D models and the
physics engine listed under **Credits** (used under their own licenses).

---

## Meta models used (all called at runtime through the Meta Model API, `https://api.meta.ai/v1`)

| Meta model | Model ID | Endpoint | What it does in Hey Robot |
|---|---|---|---|
| **Muse Voice Transcribe** | `muse-voice-transcribe-1.0` | `wss://api.meta.ai/v1/asr/realtime` | Turns the operator's spoken order into text |
| **Muse Spark** | `muse-spark-1.3-contributor` | `/v1/chat/completions` | Routes the order to a skill; writes the robot program (code as policy); confirms the product in every SAM box before a grasp; names products in the drone's close-ups; answers questions ("how many items on station C?"); compares the real camera views with the imagined goals; writes a lesson after each success |
| **Muse Wonder** | `muse-wonder-1.0` | `/v1/chat/completions` | After a failed attempt: reads the action log, the checker result and the before/after/imagined pictures, writes a diagnosis and a fixed program |
| **SAM 3.1** | `sam-3.1` | `/v1/responses` | Finds every item, the tray, the drone's landing pad and the robot's basket in camera pictures (boxes + confidence); the depth image turns each box into a 3D grasp/landing point |
| **Muse Image** | `muse-image-1.0` | `/v1/images/edits` | Imagines the finished scene before the robots act (overview + a close-up of the destination table); turns simulator pictures into realistic photos for the sim-to-real test |

No AI model runs locally, and no non-Meta AI model is used. The `openai` Python package is used only as
an HTTP client for Meta's OpenAI-compatible image endpoint.

**Meta tools used while building (not at runtime):** **Muse Code** wrote the first version of the ROS 2
bridge (`ros2_bridge.py`, `places.yaml`, `test_ros2_bridge.py`); I reviewed and fixed it (see
**Path to real robots**). Muse Spark was also used from the terminal (`ask_spark.py`) to review the project.

**Non-model libraries:** PyBullet (physics), NumPy, Pillow, requests, sounddevice + websocket-client
(microphone, voice stream), OpenCV (**only** to display the live camera window — all perception is SAM 3.1
and Muse Spark), PyTorch (only offline, to train the drone's landing reflex on Colab). The organizers
confirmed these libraries are fine.

---

## What the robots can do (9 skills + any new job, all from voice)

| Skill | Example order | Who does what |
|---|---|---|
| deliver | "I'm hungry, bring me some fruit." | Spark chooses banana, pear, strawberry; drone scouts; robot picks and puts them in the tray |
| move | "Move the banana to station C." | Robot finds, picks, drives, puts it down |
| sort | "Fruit in the tray, tools on station A." | Two groups, two destinations, one plan |
| **new job** (custom) | "Swap the banana and the pear." | No ready-made skill: Spark plans it, an independent Spark judge checks the before/after item positions |
| relay (hand-off) | "Get the strawberry from the top shelf and put it on station C." | Drone grabs it from the 1.3 m shelf (out of the arm's reach) and drops it into the robot's basket **while the robot is driving**; the arm takes it out and puts it on the table |
| air | "Drone, get the strawberry and the pear from the top shelf." | Drone vacuum gripper (≤ 100 g), SAM-guided |
| land | "Land the drone on the moving robot." | Drone lands on the moving robot in gusty wind (learned RL reflex) |
| audit | "Check the inventory and fix what is wrong." | Drone compares tables with the records, robot fixes misplaced items, missing items are reported |
| throw | "Throw a tennis ball into the blue bin." | Robot measures the throw, corrects for hidden drag |
| ask | "How many items are on station C?" | Drone flies there and looks; Spark counts and answers out loud |

---

## How it works

```mermaid
flowchart LR
    V[Voice order] -->|Muse Voice Transcribe| T[Text]
    T --> SH{SafetyShield<br/>Spark intent + fixed rules}
    SH -->|harmful| NO[Refuse out loud<br/>robot action: none]
    SH -->|unclear| Q[Ask back] --> T
    SH -->|safe| S[Skill + items + place]
    S -->|Muse Image| G[Imagined goal pictures]
    S -->|Muse Spark planner + lessons| C[Robot program]
    C --> K{Safety check<br/>AST whitelist}
    K --> R[Robots act in PyBullet]
    R -->|SAM 3.1 + Spark before every grasp| R
    R --> X[Independent checker]
    R -->|Spark: real vs imagined| G
    X -->|pass| L[Lesson saved + spoken result]
    X -->|fail| W[Muse Wonder diagnosis] --> C
```

0. **Safety first.** Every order goes through the SafetyShield (below) before anything is planned.
1. **Voice in, voice out.** Muse Voice Transcribe hears the order; the robot speaks its plan, each key
   step ("Drone: on station B I see banana and scissors", "Got the banana") and the checked result.
2. **Code as policy.** Muse Spark writes a short Python `skill(robot)` program against a documented robot
   API. An AST safety checker allows only the documented robot methods and basic Python (no imports),
   and a run stops after 60 robot actions. Lessons from earlier successes are given back to Spark as memory.
3. **Ask Muse before every action (like a Roboflow/YOLO app, but with Meta models).**
   SAM 3.1 boxes → 3D point from the depth image → Muse Spark names the product in the crop. If Spark names a
   different product, the robot tries the next box. Every detection is saved as a labelled picture and one
   JSON line in `runs/vision/` (box, confidence, 3D point, Spark's answer).
4. **Two independent judges.** A simulator checker (ground truth) decides success; Spark also compares
   the real camera views with Muse Image's imagined goal (overview + close-up). The imagination check once
   caught a can knocked off a table that the item checker missed — the checker now catches it too.
5. **Recovery.** Hidden real-world faults (grasps that slip, a worker who moves an item after the drone
   scanned it). Plans check every return value, retry slipped grasps and re-scan missing items.
6. **Team hand-off on the move.** The drone follows the robot's orange basket with SAM 3.1 (a sighting
   every 0.25 s, least-squares velocity fit, wind-drift correction), comes down while predicting where the
   basket will be, and lets go 14 cm above it. A basket camera confirms the item with SAM 3.1.
7. **Learned reflex for landing.** A PPO policy (4,096 simulated drones on a Colab T4) lands the drone on
   the moving robot in gusts; its numpy forward pass runs inside the simulation, fed by SAM 3.1 sightings.
8. **Control room.** A live 2×2 window: robot front, wrist, drone and ceiling cameras. When SAM runs, the
   tile freezes for 2 s on the labelled detection. Frames are saved in `runs/control_room/`.

---

## New jobs from one sentence (no new code)

Every new job for a normal robot means an engineer writes and tests a new program. Hey Robot has 9 ready skills,
but you can also say a job **none of them covers** — "Swap the banana and the pear", "Put one fruit on every
station", "Clear station C into the tray". The router sends it to the **new-job** path:

1. Muse Spark plans it and writes the robot program from the robot's actions. It can **compose tested skills**:
   "Swap the scissors and the medium clamp, then land the drone on the moving robot" ends with
   `land_on_moving_robot()` (the RL landing reflex).
2. Spark first predicts where every item will be afterwards, so **Muse Image draws exactly those items in exactly
   those places** (overview + close-ups of the tables that change), instead of a generic "fruit".
3. The robots act; SAM 3.1 + Spark check every item before each grasp.
4. **An independent Muse Spark judge** decides success: it sees only the order and where every item (and the
   drone) was before and after (from the simulator), never the program. On a failure, Muse Wonder diagnoses it and Spark retries.
5. The working program is saved as a lesson, so a similar job next time starts from it.

## Show and tell: point at a real object

Hold a **real** object up to the laptop webcam and say *"Bring me one of these."*
(`python command_center.py --loop --show`). The webcam picture is taken while you speak; **Muse Spark** names
what you hold and picks the matching warehouse product (a real apple → "we don't stock apples, the closest
product is the pear"); **SAM 3.1** draws its box (`runs/real/show_XX.png`); "these" in the order becomes the
product, and the order goes through the normal pipeline (SafetyShield → router → robots → checks).
It links the real world to a robot action in one sentence.

## Home mode: the same robots help people at home (`--home`)

![The robot team in home mode](docs/home_mode.png)

`python command_center.py --loop --home`. The same robot team and the same 9 skills, in a home with **two rooms**:
a living room with a **kitchen counter**, a **dining table**, a **bedside table** (grandpa resting in bed) and a
**sofa table** with the serving tray (grandma on the sofa), and a **storeroom** behind a door. Someone who cannot
walk far or reach high just says:

- *"I'm on the sofa, bring me the soup can from the storeroom."* → the drone checks the rooms, the robot drives
  through the door into the storeroom, picks the can and brings it to the serving tray
- *"Get the strawberry from the top kitchen shelf and put it on the bedside table."* → drone + robot hand-off
- *"Clear the dining table and put everything in the serving tray."* → a new job, planned by Spark

**Honest note:** the four living-room work spots keep their warehouse names inside (station A = kitchen counter,
B = dining table, C = bedside table, packing station = sofa table), so every tested move, grasp, camera and
check is reused. The storeroom table is a new 5th work spot (station D, home mode only); it sits on the diagonal
through the gap between the tables, so the robot's normal straight-line driving goes through the door. The
spoken order is translated to the inside names before Muse Spark routes it; what the robot says, the place
labels and the Muse Brain panel use the home names (the panel also shows the mapping). Walls, furniture and
people are visual only, so they can never block the robot or the drone.
With the scripted plans, all 9 skills pass in home mode, and 5/5 storeroom jobs pass (fetch from it, put into
it) **[sim mask]**. With the real models: move to the bedside table, a strawberry to the sofa, and clearing the
dining table (a new job) all passed on the first attempt **[real]**.

## The Muse Brain panel (what the video shows)

The control-room window has a **MUSE BRAIN** column next to the four cameras: the spoken order, the number of
calls to each Meta model, **the program Muse Spark wrote with the running line lit up**, and a live feed
(`[SAM 3.1] front cam 'banana': banana 0.99 | Spark: banana` → `[ROBOT] pick(banana) ok`). Every frame is saved in
`runs/control_room/` for the demo video.

## Physics: what is real in the simulation

- **Grasping:** the fingers close and must **touch the item on both finger pads, with the item between them**
  (measured contact and finger gap); otherwise the grasp fails. A flat box wider than the open gripper is
  stocked square to the table edge, and if a grip fails the robot looks again and grips with the hand turned
  90° (also when it takes an item out of its own basket). While carrying, a joint keeps the item steady, but it **breaks above 60 N** (a real slip).
- **Throwing:** real ballistics with a hidden drag force; the robot measures a test throw and corrects its speed.
- **Drone:** wind and random gusts act on the drone; the PD autopilot and the RL reflex fight them.
- **Hand-off:** the item really falls 14 cm into a basket moving at 0.2 m/s.

## SafetyShield: the robots refuse harmful orders

"Smash him with the soup can" or "throw the ball at that worker" must never reach the robots. Every spoken order
passes three layers **before** any planning (`safety.py`):

1. **Muse Spark reads the intent** (catches paraphrases and "just a prank" forms): allow / refuse harm /
   refuse damage / refuse unsafe use / **ask back** when the order is unclear ("Which item, and to which station?").
2. **Fixed rules, no AI**: a harmful verb aimed at a person or animal is refused even if Spark said allow.
   They can only add a refusal, never remove one.
3. **Hard firewall**: the robots only have whitelisted skills (scan, pick, place, carry, sort, throw a ball into
   the bin, land, answer). There is no skill that can target a person, and the AST checker rejects anything else.

If **any part** of an order is harmful, nothing is done: "Bring me this item and hit me" (with real scissors shown
to the webcam) was refused as a whole, and the robot offered "I can deliver the scissors to the tray" instead.
The robot says why it refused and offers a safe alternative; the log shows
`🛡️ SAFETY SHIELD: BLOCKED | refuse_harm | ... | robot action: none`.

Red-team test (`safety_test.py`, **[real]** Muse Spark, 25 harmful + 20 normal orders):

| Run | Harmful refused | Refused by Muse Spark alone | Normal orders allowed | False refusals |
|---|---|---|---|---|
| 1st run | **25/25** | **25/25** | 19/20 | 1/20 — "Land the drone on the moving robot" was called unsafe |
| **2nd run** (Spark told the team's built-in jobs) | **25/25** | **25/25** | **20/20** | **0/20** |

---

## Path to real robots: the ROS 2 bridge

The program Muse Spark writes calls `robot.go_to()`, `robot.pick()`, `robot.put_aside()` and so on.
`ros2_bridge.py` gives a ROS 2 robot **the same methods**, so the same Spark program can drive real hardware:

| Hey Robot call | ROS 2 side |
|---|---|
| `go_to(place)` | Nav2 `NavigateToPose` action, park poses from `places.yaml` (same poses as the simulator) |
| `pick(item)` / `place_on(thing)` / `put_aside(side)` | MoveIt 2 `MoveGroup` pose goals (hand pointing down, turned with the robot) + `GripperCommand` |
| grasp point | **SAM 3.1** box on the colour image + aligned depth + camera intrinsics (`camera_info`) + TF to the `map` frame |
| `sees`, `recall`, `is_holding`, `places` | Same behaviour as the simulator API |

- **Written with Muse Code**, then reviewed by me. Fixes made in review: real `rclpy` action calls
  (`send_goal_async` → result, with timeouts and cancel), full MoveIt pose goals (the first version sent an
  empty request), TF + intrinsics instead of a fixed camera guess, 16-bit (mm) and float (m) depth images,
  park poses and put-down spots that use the robot's heading, and the same poses as the simulator.
- **Mock mode needs no ROS:** every goal is recorded instead of sent. `python -m pytest test_ros2_bridge.py`
  runs a banana/pear swap program and checks the Nav2/MoveIt/gripper goals (11 tests pass).
- **Honest status:** ROS mode is written for a Nav2 + MoveIt 2 + Panda-style setup but has **not been run
  on a real robot**. Topic, action and link names are settings at the top of the file.

## Results (measured)

Every row says which perception it used: **[real]** = real SAM 3.1 + Muse Spark through the Meta Model API;
**[sim mask]** = the simulator's own object mask instead of SAM, used only to run many trials quickly.

| Experiment | Perception | Setup | Result |
|---|---|---|---|
| **Real-camera test** (`real_eye.py --photos`) | **[real]** | 9 phone photos of 6 real objects (apple, soup can, banana, scissors, mug, bottle) in normal / dim / cluttered light; SAM 3.1 also asked for 2 absent objects per photo | SAM 3.1 found **18/18** · Spark named **18/18** right · SAM false alarms **0/18** (every light) |
| **Sim-to-real vision test** (`vision_bench.py`) | **[real]** | 3 tables × 4 pictures: simulator, and Muse Image's realistic photo / dim light / clutter versions | SAM 3.1 found **28/28** items · Spark named **26/28** right (2 misses: an unlabeled-looking soup can) · **27/28** at the same spot on the table (IoU ≥ 0.5) |
| **Hand-off on the move** (`relay`) | **[real]** | Robot driving at 0.2 m/s, steady wind | Item dropped **1.4 cm** from the basket center, placed on the station ✅ |
| **Show and tell** (`--show`, real webcam) | **[real]** | Held up real scissors and said "Bring me this item." | Spark: "scissors" → SAM 3.1 box → order became "Bring me the scissors." → delivered to the tray ✅ first attempt (242 s) |
| **Voice session** (`command_center.py`) | **[real]** | 4 spoken orders in 2 sessions, incl. a NEW job; Spark-written programs | **4/4** first attempt (table below) |
| Hand-off sweep | [sim mask] | 4 items × 3 stations | **12/12** delivered, aim 1.3–1.5 cm |
| **Recovery from faults** (`exp_recovery.py`) | [sim mask] | 20 orders, 30 % slip chance per grasp, item moved after the scan in half the runs, **real two-finger grip** | Checking every step: **16/20** · trusting the first answer: **4/20** (with an item moved: 7/10 vs 0/10). With the older locked grip it was 19/20 vs 3/20: the real grip is harder, and we report it |
| **Real two-finger grip** | [sim mask] | Every product picked from its table | **12/12** products held with contact measured on both finger pads (gap 6–81 mm) |
| **RL landing reflex** (Colab) | simulated pad sightings | Gusts up to 1.5× the training strength | RL reflex **100 % / 2.3 cm** vs camera tracking 65.9 % / 10.1 cm (197M steps, 392 s on a T4) |
| RL landing in PyBullet (`exp_rl_landing.py`) | [sim mask] | 20 wind/gust conditions | RL **20/20** (≈1.3 cm) · tracking 13/20 · open loop 10/20 |
| All skills (`test_all_skills.py`) | [real] on your laptop, [sim mask] in CI-style offline tests | Scripted plans, same robots and checkers | **9/9** jobs pass |

![Real photos: SAM 3.1 boxes and Muse Spark names in normal, dim and cluttered light](docs/real_photos_grid.png)

The original photos are in `real_photos/`, and the per-item results are in `docs/real_eye.json`.

**Where a job's minutes go** — the same spoken sort order, real models, printed by `usage_meter.py` as `⏱️ TIMING`
(model seconds are summed per call, so parallel calls count fully):

| Run | Total | SAM 3.1 | Muse Spark | Muse Image | Physics + robot motion + other |
|---|---|---|---|---|---|
| Real-time 3D window | 1267 s | 99 calls, 266 s | 19 calls, 135 s | 2 calls, 20 s | 845 s |
| `--fast` | **616 s (−51 %)** | 99 calls, 199 s | 19 calls, 101 s | 2 calls, 35 s | 281 s |

After that measurement, two more cuts: the robot reuses its last detection when nothing changed (`sees` then
`pick`), and drone scans ask fewer alternative names. Counted on the same sort plan **[sim mask]**:
SAM 3.1 calls **89 → 60**, Spark confirmations **15 → 10**, still 9/9 jobs passing.

**Final voice session** (real Muse Voice Transcribe, Spark, Wonder on standby, SAM 3.1, Muse Image):

| Spoken order (as heard) | Routed to | Result |
|---|---|---|
| "Put the fruits on the tray and the tools in the station A." | sort (2 groups) | ✅ first attempt — 5/5 items in the right place |
| "I am hungry. Bring me some fruit." | deliver (Spark chose banana, pear, strawberry) | ✅ first attempt — imagination check MATCH |
| "How many items are on station C?" | ask | ✅ "3 items" (correct: pear, medium clamp, soup can) |
| **"Swap the banana and the pear and then land the drone on the robot."** (a NEW job) | custom — Spark wrote a program that swaps the items and ends with `land_on_moving_robot()` | ✅ first attempt — independent judge: "Banana and pear swapped positions, drone landed, all other items unchanged" |
| _(scripted run, not spoken)_ "Get the strawberry from the top shelf and put it on station C." | relay (hand-off) | ✅ dropped 1.4 cm from the basket center while driving (`test_all_skills.py relay`) |

---

## Run it

```powershell
python -m venv venv; venv\Scripts\activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/eleramp/pybullet-object-models third_party/pybullet-object-models
git clone --depth 1 https://github.com/utiasDSL/gym-pybullet-drones third_party/gym-pybullet-drones
setx MODEL_API_KEY "<your Meta Model API key>"     # then open a new terminal

python command_center.py --loop                    # speak orders (6 s each); the robots act and talk back
python command_center.py --text "Put the fruit in the tray and the tools on station A."
python test_all_skills.py                          # all 9 skills with scripted plans
python test_all_skills.py relay                    # one skill
python vision_bench.py                             # sim-to-real test with Muse Image + SAM 3.1 + Spark
python real_eye.py --photos real_photos            # REAL photos: SAM 3.1 + Spark find and name real objects
python real_eye.py --webcam                        # live webcam: ask by voice, Spark + SAM answer out loud
python command_center.py --loop --show             # show and tell: hold up a real object, "bring me one of these"
python command_center.py --loop --home             # home mode: two rooms (living room + storeroom), people
python safety_test.py                              # red-team test: 25 harmful + 20 normal orders
python ask_spark.py --file README.md "question"    # ask Muse Spark about the project
python -m pytest test_ros2_bridge.py               # ROS 2 bridge in mock mode (no ROS needed)
```

Options: `--quiet` (robot does not speak), `--fast` (no real-time wait in the 3D window), `--faults` (hidden slips and moved items),
`--manual` (scripted plan instead of Spark's program), `--home` (home mode). Set `SKILLFORGE_NO_WINDOW=1` to hide the camera window.

## Files

| File | What it is |
|---|---|
| `command_center.py` | Voice loop: hear → route (Spark) → run the skill → speak the result |
| `learner.py` | Imagine (Muse Image) → plan (Spark) → act → check → diagnose (Wonder) → lesson |
| `team_skills.py` | The robot API that Spark programs against (drone + ground robot + hand-off) |
| `skill_tasks.py` | The 9 skills: scenes, checkers, goal texts, scripted reference plans |
| `mobile_env.py` | Husky + Panda robot, cameras, perception, grasping, the basket |
| `drone.py` | Drone autopilot, gusts, SAM-guided landing (tracking + RL reflex), vacuum gripper, moving drop |
| `muse_eye.py` | Vision log (labelled pictures + JSONL), Spark confirmation, the control room |
| `store_tasks.py`, `mission_tasks.py`, `audit_tasks.py`, `throw_skills.py`, `ask_tasks.py` | Scenes and skills |
| `perception.py`, `llm.py`, `imagine.py`, `voice.py` | Thin clients for SAM 3.1, Muse Spark/Wonder, Muse Image, Muse Voice Transcribe |
| `speech.py` | The robot talks back (Windows built-in voice) |
| `home.py` | Home mode (`--home`): two rooms, people, the storeroom, and the home names for the same robots and skills |
| `real_eye.py` | The same SAM 3.1 + Spark pipeline on real phone photos and a live webcam |
| `show_tell.py` | Show and tell: a real object held up to the webcam becomes the item in a voice order |
| `usage_meter.py` | Counts every Meta model call and its time per job |
| `safety.py`, `safety_test.py` | SafetyShield (Spark intent check + fixed rules + firewall) and its red-team test |
| `vision_bench.py`, `exp_recovery.py`, `exp_rl_landing.py`, `colab_rl.py` | Experiments and RL training |
| `ros2_bridge.py`, `places.yaml`, `test_ros2_bridge.py` | ROS 2 bridge with the same robot API (Nav2 + MoveIt 2 + SAM 3.1), written with Muse Code and reviewed; mock-tested |

## Limitations (honest)

- The robots are simulated. The perception pipeline is tested on real phone photos and a live webcam
  (`real_eye.py`), but no real robot hardware was used.
- The ROS 2 bridge is tested only in mock mode (no ROS install, no robot).
- Each SAM/Spark call goes over the internet, so a full voice job takes a few minutes.
- The drone gripper is limited to light items (≤ 100 g); very wide items (a chips can) do not fit the gripper.
- After a two-finger grip is confirmed, a joint (that breaks above 60 N) holds the item while carrying, instead
  of simulating finger friction alone, which is unstable in PyBullet.
- Some sweeps (hand-off 12/12, recovery, PyBullet landing) used the simulator's object mask instead of SAM to
  run many trials quickly; the headline runs above used the real Meta models.

## Credits and licenses (third-party assets, not built during the hackathon)

| Asset | Source | License |
|---|---|---|
| Physics engine and robot models (Husky, Franka Panda, kiva shelf, plane) | PyBullet / `pybullet_data` | zlib (models under their original licenses) |
| YCB object models (banana, pear, strawberry, scissors, clamp, cans, boxes) | [pybullet-object-models](https://github.com/eleramp/pybullet-object-models) (YCB dataset) | LGPL-2.1 (repo), YCB models CC BY 4.0 |
| Crazyflie drone model (`cf2x.urdf`) | [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) | MIT |

These two repositories are not copied into this repo: the **Run it** steps clone them into `third_party/`, with their original LICENSE files.

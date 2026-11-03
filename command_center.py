"""Hey Robot command center: say what you want, and the robot team does it.
  Muse Voice Transcribe hears the order -> Muse Spark picks the skill -> Muse Spark writes the plan
  -> the robots act (SAM 3.1 eyes) -> an independent checker confirms -> Wonder diagnoses failures.

python command_center.py                        # speak for 6 s
python command_center.py --seconds 8            # speak for 8 s
python command_center.py --text "throw the ball into the bin"     # typed order (no microphone)
python command_center.py --manual               # Spark routes the order, a scripted plan acts (fast)
python command_center.py --loop                 # keep taking orders, one after another
python command_center.py --faults               # real-world faults on: grasps slip, a worker moves an item
"""
import json
import os
import re
import sys

import pybullet as p

from skill_tasks import ITEMS, PLACES, run_plan, task_for
from speech import say
from team_skills import MAP_FILE

ARGS = sys.argv[1:]
MANUAL = "--manual" in ARGS
FAULTS = "--faults" in ARGS          # real-world faults: slipping grasps, a worker moves an item
LOOP = "--loop" in ARGS
SHOW = "--show" in ARGS              # show and tell: hold up a real object to the webcam ("bring me one of these")
if "--fast" in ARGS:                 # the 3D window runs as fast as it can (no real-time wait)
    os.environ["SKILLFORGE_FAST"] = "1"
SKILLS = {"deliver": "bring one or more items to the packing station and put them in the tray",
          "move": "move ONE item to another station (station A, B or C)",
          "throw": "throw a tennis ball into the blue bin",
          "audit": "check the inventory against the records, fix misplaced items, report missing ones",
          "land": "land the drone on the moving ground robot (gusty wind)",
          "sort": "sort several items into groups, each group into the tray or onto a station's table",
          "ask": "answer a QUESTION: what or how many items are on a station (give place), or where an item is "
                 "(give items); the drone goes and looks",
          "air": "the DRONE fetches 1 or 2 light items (banana, pear, strawberry, scissors, medium clamp) from the "
                 "high top shelf into the tray",
          "relay": "TEAM HAND-OFF: ONE light item from the high top shelf goes onto a STATION's table (give place): "
                   "the drone grabs it and drops it into the moving robot's basket, the arm puts it on the table",
          "custom": "a NEW job none of the skills above covers, that the robots can still do by carrying items "
                    "between the stations, the tray and the top shelf (for example: swap two items, put one fruit "
                    "on every station, clear a station into the tray, move everything back), also when it ends with "
                    "landing the drone on the moving robot. List the items it uses."}
# words the transcript may contain (including common mishearings) -> item names the robots know
KEYWORDS = {"banana": "banana", "soup": "soup can", "soap": "soup can", "mustard": "mustard bottle",
            "brick": "foam brick", "gelatin": "gelatin box", "jello": "gelatin box", "pear": "pear",
            "strawberr": "strawberry", "scissor": "scissors", "clamp": "medium clamp"}
ROUTER = f"""You route spoken orders for a warehouse robot team: a drone and a wheeled robot with an arm.
The text comes from speech recognition and may contain small mistakes (for example "soap can" means
"soup can", "parking station" means "packing station", and "through the ball" means "throw the ball").
Be generous: if the order is close to a skill, choose that skill.
Pick exactly ONE skill:
{chr(10).join(f"- {k}: {v}" for k, v in SKILLS.items())}
Known items: {", ".join(ITEMS)}. Known places: {", ".join(PLACES)}.
The operator may name a CATEGORY instead of items ("some fruit", "something to eat", "the tools",
"something to cut paper with"): use common sense to choose the matching known items.
Reply with JSON only, for example:
{{"skill": "deliver", "items": ["banana", "soup can"], "place": null, "reply": "Bringing the banana and the soup can to the tray."}}
{{"skill": "move", "items": ["banana"], "place": "station C", "reply": "Moving the banana to station C."}}
{{"skill": "sort", "groups": [{{"items": ["banana", "pear"], "to": "tray"}}, {{"items": ["scissors"], "to": "station A"}}], "reply": "Fruit to the tray, tools to station A."}}
{{"skill": "ask", "items": [], "place": "station C", "reply": "Let me send the drone to look."}}
{{"skill": "relay", "items": ["strawberry"], "place": "station C", "reply": "The drone will hand the strawberry to the robot on the move."}}
{{"skill": "custom", "items": ["banana", "pear"], "place": null, "reply": "A new job: I will plan how to swap the banana and the pear."}}
{{"skill": "custom", "items": ["scissors", "medium clamp"], "place": null, "reply": "A new job: swap them, then land the drone on the moving robot."}}
If the order fits no skill: {{"skill": "none", "items": [], "place": null, "reply": "<short reason>"}}"""


def get_order():
    if SHOW:                                       # aim the real object at the webcam before speaking
        try:
            import show_tell
            show_tell.preview(2.5)
        except Exception as e:
            print(f"   (webcam unavailable: {e})")
    if "--text" in ARGS:
        i = ARGS.index("--text")
        return " ".join(a for a in ARGS[i + 1:] if not a.startswith("--"))
    from voice import listen
    secs = int(ARGS[ARGS.index("--seconds") + 1]) if "--seconds" in ARGS else 9    # room for 2-part orders
    return listen(secs)


def items_in(text):
    low, out = text.lower(), []
    for word, item in KEYWORDS.items():
        if word in low and item not in out:
            out.append(item)
    return out


def keyword_route(order):
    low = order.lower()
    place = next((pl for pl in PLACES if pl in low or f"station {pl[-1].lower()}" in low), None)
    items = items_in(order)
    if low.startswith(("how many", "what is on", "what's on", "where is", "where's")) or low.endswith("?"):
        skill = "ask"
    elif any(w in low for w in ("throw", "ball", "bin", "basket")):
        skill = "throw"
    elif "shelf" in low and items and place:
        skill = "relay"
    elif "shelf" in low and items:
        skill = "air"
    elif any(w in low for w in ("audit", "inventory", "check the")):
        skill = "audit"
    elif any(w in low for w in ("land", "drone")) and not items:
        skill = "land"
    elif items and place and any(w in low for w in ("move", "put", "take", "place")):
        skill = "move"
    elif items:
        skill = "deliver"
    else:
        return {"skill": "none", "items": [], "place": None, "reply": "I did not understand the order.",
                "by": "keywords"}
    return {"skill": skill, "items": items, "place": place, "reply": "", "by": "keywords"}


def route(order):
    """Muse Spark turns the spoken order into ONE skill call. Keywords are only the fallback."""
    try:
        from llm import spark
        text, _ = spark([{"role": "system", "content": ROUTER}, {"role": "user", "content": order}])
        cmd = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        skill = str(cmd.get("skill", "")).lower()
        items = [str(i).lower() for i in (cmd.get("items") or [])]
        items = [i if i in ITEMS else (items_in(i) or [None])[0] for i in items]
        items = [i for i in dict.fromkeys(items) if i]
        place = cmd.get("place")
        place = place if place in PLACES else None
        if skill == "sort":
            groups = []
            for g in cmd.get("groups") or []:
                its = [str(i).lower() for i in (g.get("items") or [])]
                its = [i for i in dict.fromkeys(i if i in ITEMS else (items_in(i) or [None])[0] for i in its) if i]
                to = str(g.get("to", "")).strip()
                to = "tray" if "tray" in to.lower() else next((pl for pl in PLACES if pl.lower() == to.lower()), None)
                if its and to:
                    groups.append({"items": its, "to": to})
            if groups:
                return {"skill": "sort", "items": [i for g in groups for i in g["items"]], "place": None,
                        "groups": groups, "reply": str(cmd.get("reply", "")), "by": "Muse Spark"}
        elif skill == "ask" and (place or items):
            return {"skill": "ask", "items": items, "place": place, "reply": str(cmd.get("reply", "")),
                    "by": "Muse Spark"}
        elif skill in SKILLS and skill != "ask" and (skill not in ("deliver", "move") or items) and (skill != "move" or place):
            return {"skill": skill, "items": items, "place": place, "reply": str(cmd.get("reply", "")),
                    "by": "Muse Spark"}
        if skill == "none":
            kw = keyword_route(order)             # second opinion for misheard orders
            if kw["skill"] != "none":
                kw["by"] = "keywords (Spark was unsure)"
                return kw
            return {"skill": "none", "items": [], "place": None, "reply": str(cmd.get("reply", "")),
                    "by": "Muse Spark"}
        print(f"   (Spark's answer was incomplete: {cmd}; using keywords)")
    except Exception as e:
        print(f"   (Spark router unavailable: {type(e).__name__}: {str(e)[:120]}; using keywords)")
    return keyword_route(order)


def library_for(skill):
    """Each skill keeps its own lesson memory. New ones start from the newest earlier lessons."""
    path = f"lessons_{skill}.json"
    if not os.path.exists(path):
        lib = []
        if skill in ("deliver", "move", "audit", "land"):
            for f in ["mission_library.json", "landing_library.json"]:
                if os.path.exists(f):
                    lib += json.load(open(f))[-1:]
        json.dump(lib, open(path, "w"), indent=2)
    return path


def run(cmd, order):
    if cmd["skill"] == "ask":                    # a question: the drone looks, Spark answers
        from ask_tasks import run_ask
        return run_ask(order, cmd["place"], cmd["items"]), 1
    if cmd["skill"] == "custom" and cmd.get("reply"):
        # speech recognition can mishear ("beer" for "pear"): the planner and the judge get Spark's cleaned meaning
        order = f'{order} -> meaning: {cmd["reply"]} (items: {", ".join(cmd["items"]) or "see the order"})'
    t = task_for(cmd["skill"], order, cmd["items"], cmd["place"], faults=FAULTS, groups=cmd.get("groups"))
    if os.path.exists(MAP_FILE):
        os.remove(MAP_FILE)
    if MANUAL:                                   # scripted plan: no Spark planning, same robots and checker
        env = t["scene"](gui=True)
        api = t["api_cls"](env)
        try:
            run_plan(cmd["skill"], api, cmd["items"], cmd["place"], groups=cmd.get("groups"))
        except Exception as e:
            print(f"   plan crashed: {type(e).__name__}: {e}")
        env.step(240)
        ok, msg = t["check"](env)
        for s in api.log:
            print("   ", s)
        print(f"CHECK: {ok} - {msg}")
        env.say("DONE!" if ok else "NOT DONE")
        env.step(720)                            # hold the final shot
        p.disconnect()
        return ok, 1
    from learner import learn
    r = learn(t["task"], t["scene"], t["check"], library=library_for(cmd["skill"]), api_cls=t["api_cls"],
              api_doc=t["api_doc"], methods=t["methods"], imagine_text=t["goal"],
              use_imagination=t["goal"] is not None, extra_goals=t.get("extra_goals", ()))
    return r["success"], r["attempts"]


def show_and_tell(order):
    """Webcam picture taken right after you spoke -> Spark names the object -> SAM boxes it -> the order now
    names the product. Returns the new order, or None when there is nothing to fetch."""
    try:
        import show_tell
        frames = [show_tell.last_preview(), show_tell.grab()]   # held up before speaking, and after
        new, line, info = show_tell.resolve(order, frames)
    except Exception as e:
        print(f"   (show and tell unavailable: {type(e).__name__}: {str(e)[:80]}; using the order as heard)")
        return order
    print(f"📷 SHOW AND TELL: Spark sees '{info['object']}' -> product: {info['product']} "
          f"| SAM 3.1 box: {info['box']} | picture: {info['picture']}")
    print(f"🤖 {line}")
    say(line, wait=True)
    if new:
        print(f'🗣️  ORDER NOW: "{new}"')
    return new


def shield(order):
    """SafetyShield before planning (see safety.py). Returns (order, "allow" or "refuse").
    If the order is unclear, the robot asks ONE question and adds the answer to the order."""
    from safety import check
    for _ in range(2):
        r = check(order)
        try:
            import muse_eye
            muse_eye.event("SAFETY", f"{r['verdict'].upper()} ({r['category']}) by {r['by'][:30]}")
        except Exception:
            pass
        if r["verdict"] == "refuse":
            print(f"🛡️  SAFETY SHIELD: BLOCKED | {r['category']} | {r['reason']} | decided by {r['by']} | "
                  "robot action: none")
            print(f"🤖 {r['reply']}")
            say(r["reply"], wait=True)
            return order, "refuse"
        if r["verdict"] != "clarify":
            return order, "allow"
        print(f"❓ CLARIFY ({r['by']}): {r['reply']}")
        say(r["reply"], wait=True)
        try:
            answer = input("Type the answer: ").strip() if "--text" in ARGS else get_order().strip()
        except Exception:
            answer = ""
        print(f'🗣️  ANSWER: "{answer}"')
        if not answer:
            return order, "allow"
        order = f"{order} ({answer})"
    return order, "allow"


def main():
    if not os.environ.get("MODEL_API_KEY"):
        print("❌ MODEL_API_KEY is not set. Get a key at dev.meta.ai, then run:\n"
              '   setx MODEL_API_KEY "<your key>"   and open a NEW terminal (or: $env:MODEL_API_KEY="<your key>")')
        return
    import usage_meter
    usage_meter.install()                          # count every Meta model call and its time
    history = []
    while True:
        try:
            order = get_order().strip()
        except Exception as e:                     # e.g. the hotspot dropped the voice connection
            print(f"\n⚠️  Voice failed: {type(e).__name__}: {str(e)[:100]}. Check the hotspot, then try again.")
            say("I could not hear you. Please check the internet and try again.")
            if not LOOP or input("Press Enter to try again, or type q to quit: ").strip().lower() == "q":
                break
            continue
        print(f'\n🗣️  HEARD: "{order}"')
        if SHOW and order:
            order = show_and_tell(order)
            if order is None:
                if not LOOP or "--text" in ARGS:
                    break
                if input("\nPress Enter for the next order, or type q to quit: ").strip().lower() == "q":
                    break
                continue
        try:
            import muse_eye
            muse_eye.set_order(order)
        except Exception:
            pass
        if not order:
            print('Nothing was heard. Start right after "Speak now", or use --text "your order".')
        else:
            order, verdict = shield(order)          # SafetyShield: refuse harmful orders, ask back if unclear
            if verdict == "refuse":
                history.append((order, "REFUSED", True, 0))
                if not LOOP or "--text" in ARGS:
                    break
                if input("\nPress Enter for the next order, or type q to quit: ").strip().lower() == "q":
                    break
                continue
            cmd = route(order)
            print(f"🧭 ROUTED by {cmd['by']}: skill={cmd['skill']} items={cmd['items']} place={cmd['place']}"
                  + (f" groups={cmd['groups']}" if cmd.get("groups") else ""))
            if cmd["reply"]:
                print(f"🤖 {cmd['reply']}")
                say(cmd["reply"], important=True)
            elif cmd["skill"] == "none":
                say("Sorry, I did not understand that order.", important=True)
            if cmd["skill"] != "none":
                usage_meter.start()
                ok, attempts = run(cmd, order)
                usage_meter.report(cmd["skill"])
                history.append((order, cmd["skill"], ok, attempts))
                print(f"\n=== RESULT: {cmd['skill']} -> success={ok} (attempts={attempts}) ===")
                if cmd["skill"] != "ask":
                    say("Done. The job is finished and checked." if ok else
                        "Sorry, I could not finish the job. The checker says it is not done.", wait=True)
        if not LOOP or "--text" in ARGS:
            break
        if input("\nPress Enter for the next order, or type q to quit: ").strip().lower() == "q":
            break
    if history:
        print("\n=========== SESSION SUMMARY ===========")
        for order, skill, ok, attempts in history:
            tag = "SAFE" if skill == "REFUSED" else ("PASS" if ok else "FAIL")
            print(f'{tag}  {skill:8s} attempts={attempts}  "{order}"')


if __name__ == "__main__":
    main()

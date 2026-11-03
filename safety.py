"""SafetyShield: the robots refuse harmful orders, and ask back when an order is unclear.

Three layers:
  1. Muse Spark reads the INTENT of every spoken order (catches paraphrases and jokes, not just keywords):
     allow / refuse_harm / refuse_damage / refuse_unsafe / clarify.
  2. Fixed rules (no AI): a harmful verb aimed at a person or animal is refused even if Spark said "allow".
     These rules can only ADD a refusal, never remove one.
  3. Hard firewall (in the code): the robots only have whitelisted skills (scan, pick, place, carry, sort,
     throw a ball INTO THE BIN, land, answer). There is no skill that can target a person, and the AST
     safety checker rejects any other call in Spark's program.

check(order) -> {"verdict": "allow" | "refuse" | "clarify", "category", "reason", "reply", "by"}
"""
import json
import re

SAFETY = """You are the safety officer of a warehouse robot team: a drone and a wheeled robot with an arm.
These are the team's NORMAL, tested built-in jobs - always allow them:
deliver items to the tray; move or put an item on a station; sort items into groups; the drone fetches light
items from the high top shelf; the drone hands an item to the robot while the robot drives; throw a tennis
ball into the blue bin; LAND THE DRONE ON THE MOVING ROBOT (its normal docking skill, done every day);
check the inventory and fix misplaced items (an audit: the robots know the records, no details needed);
answer questions about what is where.
The text comes from speech recognition and may contain small mistakes. Classify the operator's order:
- allow: one of the jobs above or another normal warehouse task, even if it is phrased casually or uses
  words like "quickly" or "beat the clock".
- refuse_harm: it could hurt or scare a person or an animal: hitting, throwing or dropping things at or on
  someone, chasing, crashing into, pushing, trapping - including indirect, joking or "just pretend" forms.
- refuse_damage: deliberately breaking, smashing or destroying things or property, or sabotage.
- refuse_unsafe: unsafe use of the machines: switching off safety checks, flying over people's heads,
  carrying people, going faster than allowed, lifting things far too heavy.
- clarify: the robot truly cannot act because something needed is missing (for example "put it there"
  with no item and no place). Categories such as "some fruit" or "the tools" are clear enough.
  Speech recognition often mishears words: if you can guess the intended normal job ("svep" or "swab" = "swap",
  "parking station" = "packing station"), choose allow, not clarify.
Reply with JSON only:
{"verdict": "allow|refuse_harm|refuse_damage|refuse_unsafe|clarify", "reason": "<a few words>",
 "reply": "<ONE short sentence the robot will SAY. For a refusal: say you cannot do that because it could hurt
 someone (or damage things / is unsafe) and offer a safe warehouse alternative. For clarify: ask one short
 question. For allow: empty>"}"""

HARM_VERBS = (r"hit|smash|beat|beat up|punch|kick|slap|stab|kill|hurt|injure|attack|strike|shoot|crush|"
              r"push|shove|chase|scare|trip|knock|throw|toss|hurl|drop|fling|crash|ram|fly into|run over|"
              r"poke|whack|bash|harm|squash")
TARGETS = (r"him|her|them|me|us|you|someone|somebody|anyone|people|person|persons|man|men|woman|women|guy|"
           r"guys|boy|girl|kid|kids|child|children|baby|worker|workers|colleague|boss|manager|operator|"
           r"human|humans|customer|dog|cat|animal|bird|face|head|his|their|my")
DAMAGE = r"(smash|destroy|break|wreck|burn|set fire to|sabotage)\s+(the\s+)?(robot|drone|shelf|shelves|table|window|door|camera|everything|it all)"
UNSAFE = (r"(turn|switch)\s+off\s+(the\s+)?safety|disable\s+(the\s+)?safety|ignore\s+(the\s+)?safety|"
          r"over (their|people'?s|his|her) heads|\bcarry (me|him|her|them|someone|people|a person)\b")


def rule_check(order):
    """Layer 2: fixed rules. Returns (category, reason) or None."""
    low = " " + order.lower() + " "
    if re.search(rf"\b({HARM_VERBS})\b[^.!?]{{0,40}}\b({TARGETS})\b", low) and \
            not re.search(r"\b(into|in) the (blue )?bin\b", low):
        return "refuse_harm", "a harmful action aimed at a person or animal"
    if re.search(rf"\b(fly|drive|steer|send|move|aim)\b[^.!?]{{0,25}}\b(into|at|towards?)\b[^.!?]{{0,25}}\b({TARGETS})\b", low) \
            or re.search(r"\bteach\b[^.!?]{0,30}\blesson\b", low):
        return "refuse_harm", "aiming a machine at a person"
    if re.search(DAMAGE, low):
        return "refuse_damage", "deliberate damage"
    if re.search(UNSAFE, low):
        return "refuse_unsafe", "unsafe use of the machines"
    return None


REFUSE_REPLY = {"refuse_harm": "I can't do that, it could hurt someone. I can move, sort or deliver items instead.",
                "refuse_damage": "I can't do that, it would damage things. I can tidy or sort the items instead.",
                "refuse_unsafe": "I can't do that, it is not safe. My safety checks always stay on."}


def check(order):
    """All three layers. Never raises: if Spark is unavailable, the fixed rules still decide."""
    verdict, reason, reply, by = "allow", "", "", "Muse Spark"
    try:
        from llm import spark
        text, _ = spark([{"role": "system", "content": SAFETY}, {"role": "user", "content": order}], effort="low")
        j = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        verdict = str(j.get("verdict", "allow")).lower().strip()
        reason, reply = str(j.get("reason", "")), str(j.get("reply", ""))
        if verdict not in ("allow", "refuse_harm", "refuse_damage", "refuse_unsafe", "clarify"):
            verdict = "allow"
    except Exception as e:
        by = f"fixed rules (Spark unavailable: {type(e).__name__})"
    rule = rule_check(order)
    if rule and not verdict.startswith("refuse"):           # rules can only add a refusal
        verdict, reason, by = rule[0], rule[1], "fixed rules (Spark said allow)" if by == "Muse Spark" else by
        reply = ""
    if verdict.startswith("refuse"):
        return {"verdict": "refuse", "category": verdict, "reason": reason,
                "reply": reply or REFUSE_REPLY[verdict], "by": by}
    if verdict == "clarify":
        return {"verdict": "clarify", "category": "clarify", "reason": reason,
                "reply": reply or "Sorry, which item and which place do you mean?", "by": by}
    return {"verdict": "allow", "category": "allow", "reason": reason, "reply": "", "by": by}

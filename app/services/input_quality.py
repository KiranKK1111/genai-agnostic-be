"""Input quality gate — gibberish and nonsense detection with 6-signal scoring."""
import re
import math
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# Common English words for dictionary check
COMMON_WORDS = set("the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us".split())
KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _char_entropy(text: str) -> float:
    """Signal 1: Shannon entropy per character."""
    if not text:
        return 0
    freq = Counter(text.lower())
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)

def _dict_word_ratio(text: str) -> float:
    """Signal 2: Percentage of tokens matching common dictionary."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0
    return sum(1 for w in words if w in COMMON_WORDS) / len(words)

def _keyboard_adjacency(text: str) -> float:
    """Signal 5: Percentage of adjacent keyboard character pairs."""
    text = text.lower()
    if len(text) < 2:
        return 0
    adjacent = 0
    for i in range(len(text) - 1):
        for row in KEYBOARD_ROWS:
            idx_a = row.find(text[i])
            idx_b = row.find(text[i + 1])
            if idx_a >= 0 and idx_b >= 0 and abs(idx_a - idx_b) <= 1:
                adjacent += 1
                break
    return adjacent / (len(text) - 1)

def _repeated_chars(text: str) -> float:
    """Signal 4: Longest consecutive same-character run."""
    if not text:
        return 0
    max_run = 1
    current = 1
    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return min(max_run / 5.0, 1.0)

def _vowel_ratio(text: str) -> float:
    """Signal 3: Vowel/consonant ratio."""
    alpha = re.findall(r"[a-zA-Z]", text)
    if not alpha:
        return 0.5
    vowels = sum(1 for c in alpha if c.lower() in "aeiou")
    return vowels / len(alpha)

def _language_confidence(text: str) -> float:
    """Signal 6: Language detection confidence via langdetect."""
    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if results:
            return results[0].prob
    except Exception:
        pass
    return 0.0


def detect_gibberish(text: str) -> dict:
    """Score input quality using 6-signal weighted composite.
    Returns {score, is_gibberish, tier, reason}.
    Tiers: 'pass' (<0.4), 'rephrase' (0.4-0.7), 'reject' (>0.7)."""
    if not text or not text.strip():
        return {"score": 1.0, "is_gibberish": True, "tier": "reject", "reason": "Empty input"}

    text = text.strip()

    # Short greetings are always OK
    if len(text) <= 10 and text.lower() in ["hi", "hello", "hey", "help", "thanks", "bye", "ok", "yes", "no"]:
        return {"score": 0.0, "is_gibberish": False, "tier": "pass", "reason": "Known greeting"}

    # Calculate all 6 signals
    entropy = _char_entropy(text)
    dict_ratio = _dict_word_ratio(text)
    vowel = _vowel_ratio(text)
    repeat = _repeated_chars(text)
    kbd_adj = _keyboard_adjacency(text)

    # Signal 6: language detection (only for longer inputs to avoid noise)
    lang_conf = 0.0
    if len(text) > 15:
        lang_conf = _language_confidence(text)

    # Weighted composite score (weights from architecture spec)
    score = 0.0
    if entropy > 4.5:
        score += 0.25       # Signal 1 weight
    if dict_ratio < 0.3:
        score += 0.30       # Signal 2 weight
    if vowel < 0.15 or vowel > 0.75:
        score += 0.10       # Signal 3 weight
    if repeat > 0.5:
        score += 0.10       # Signal 4 weight
    if kbd_adj > 0.6:
        score += 0.15       # Signal 5 weight
    if len(text) > 15 and lang_conf < 0.4:
        score += 0.10       # Signal 6 weight

    score = round(score, 3)

    # 3-tier classification
    if score > 0.7:
        tier = "reject"
    elif score > 0.4:
        tier = "rephrase"
    else:
        tier = "pass"

    return {
        "score": score,
        "is_gibberish": score > 0.4,
        "tier": tier,
        "reason": f"entropy={entropy:.2f} dict={dict_ratio:.2f} vowel={vowel:.2f} repeat={repeat:.2f} kbd={kbd_adj:.2f} lang={lang_conf:.2f}",
    }

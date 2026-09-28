"""Generate a Sinhala YouTube content match/mismatch CSV and quality report."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.data_utils import clean_text


OUTPUT_COLUMNS = [
    "video_id",
    "group_id",
    "title",
    "description",
    "transcript",
    "label",
    "topic",
    "hard_negative_type",
]

Scenario = tuple[str, str, str, str, str]
SCENARIO_FIELDS = ("topic", "subject", "claim", "source", "place")


@dataclass(frozen=True)
class HardNegativeCase:
    """A close mismatch where only the declared important fact changes."""

    change_type: str
    metadata: Scenario
    transcript: Scenario
    changed_fields: tuple[str, ...]


HARD_NEGATIVE_CHANGE_TYPES = {
    "person",
    "location",
    "product",
    "number",
    "event",
    "main_topic",
    "result_or_outcome",
    "health_or_financial_claim",
    "time_period",
}


HARD_NEGATIVE_CASES = [
    HardNegativeCase(
        "person",
        ("Politics", "අයවැය බදු යෝජනාව", "යෝජනාව සිකුරාදා ඉදිරිපත් කරනවා", "මුදල් අමාත්‍යවරයා", "පාර්ලිමේන්තුව"),
        ("Politics", "අයවැය බදු යෝජනාව", "යෝජනාව සිකුරාදා ඉදිරිපත් කරනවා", "විපක්ෂ මන්ත්‍රීවරයා", "පාර්ලිමේන්තුව"),
        ("source",),
    ),
    HardNegativeCase(
        "person",
        ("Education", "O/L ප්‍රතිඵල නිවේදනය", "ප්‍රතිඵල සඳුදා නිකුත් කරනවා", "විභාග කොමසාරිස්වරයා", "කොළඹ"),
        ("Education", "O/L ප්‍රතිඵල නිවේදනය", "ප්‍රතිඵල සඳුදා නිකුත් කරනවා", "ගුරු සංගමයේ සභාපති", "කොළඹ"),
        ("source",),
    ),
    HardNegativeCase(
        "location",
        ("Weather", "තද වැසි අනතුරු ඇඟවීම", "රාත්‍රියේ තද වැසි ඇතිවිය හැකියි", "කාලගුණ අංශය", "කොළඹ දිස්ත්‍රික්කය"),
        ("Weather", "තද වැසි අනතුරු ඇඟවීම", "රාත්‍රියේ තද වැසි ඇතිවිය හැකියි", "කාලගුණ අංශය", "ගාල්ල දිස්ත්‍රික්කය"),
        ("place",),
    ),
    HardNegativeCase(
        "location",
        ("Agriculture", "පොහොර බෙදාහැරීම", "ලබන සඳුදා බෙදාහැරීම ආරම්භ වෙනවා", "කෘෂිකර්ම නිලධාරියා", "අනුරාධපුර"),
        ("Agriculture", "පොහොර බෙදාහැරීම", "ලබන සඳුදා බෙදාහැරීම ආරම්භ වෙනවා", "කෘෂිකර්ම නිලධාරියා", "පොළොන්නරුව"),
        ("place",),
    ),
    HardNegativeCase(
        "product",
        ("Mobile phones", "iPhone 17", "camera සහ battery performance review කරනවා", "tech channel", "online"),
        ("Mobile phones", "Samsung Galaxy S26", "camera සහ battery performance review කරනවා", "tech channel", "online"),
        ("subject",),
    ),
    HardNegativeCase(
        "product",
        ("Vehicles", "Toyota Aqua", "2026 used car price review කරනවා", "vehicle reviewer", "කොළඹ"),
        ("Vehicles", "Honda Vezel", "2026 used car price review කරනවා", "vehicle reviewer", "කොළඹ"),
        ("subject",),
    ),
    HardNegativeCase(
        "number",
        ("Finance", "නිවාස ණය පොලිය", "වාර්ෂික පොලිය 8% දක්වා අඩු කළා", "බැංකු නිවේදනය", "ශ්‍රී ලංකාව"),
        ("Finance", "නිවාස ණය පොලිය", "වාර්ෂික පොලිය 12% දක්වා වැඩි කළා", "බැංකු නිවේදනය", "ශ්‍රී ලංකාව"),
        ("claim",),
    ),
    HardNegativeCase(
        "number",
        ("Crime", "පොලිස් අත්අඩංගුවට ගැනීම", "සැකකරුවන් දෙදෙනෙක් අත්අඩංගුවට ගත්තා", "පොලිස් මාධ්‍ය ඒකකය", "කුරුණෑගල"),
        ("Crime", "පොලිස් අත්අඩංගුවට ගැනීම", "සැකකරුවන් පස්දෙනෙක් අත්අඩංගුවට ගත්තා", "පොලිස් මාධ්‍ය ඒකකය", "කුරුණෑගල"),
        ("claim",),
    ),
    HardNegativeCase(
        "event",
        ("Sports", "අවසන් මහා ක්‍රිකට් තරගය", "හෙට සවස ආරම්භ වෙනවා", "ක්‍රීඩා වාර්තාව", "කොළඹ ක්‍රීඩාංගණය"),
        ("Sports", "පුහුණු ක්‍රිකට් තරගය", "හෙට සවස ආරම්භ වෙනවා", "ක්‍රීඩා වාර්තාව", "කොළඹ ක්‍රීඩාංගණය"),
        ("subject",),
    ),
    HardNegativeCase(
        "event",
        ("Technology", "නව app launch උත්සවය", "සිකුරාදා සජීවීව පැවැත්වෙනවා", "software company", "online"),
        ("Technology", "developer meetup වැඩසටහන", "සිකුරාදා සජීවීව පැවැත්වෙනවා", "software company", "online"),
        ("subject",),
    ),
    HardNegativeCase(
        "main_topic",
        ("Technology", "Python වලින් mobile app එකක් හදන පාඩම", "Kivy භාවිතයෙන් app screen එකක් හදනවා", "software trainer", "online"),
        ("Technology", "Python variables පාඩම", "variables සහ data types හඳුන්වා දෙනවා", "software trainer", "online"),
        ("subject", "claim"),
    ),
    HardNegativeCase(
        "main_topic",
        ("Education", "A/L ගණිත integration පාඩම", "integration ප්‍රශ්න විසඳන ක්‍රම පෙන්වනවා", "ගණිත ගුරුවරයා", "online class"),
        ("Education", "A/L ගණිත matrices පාඩම", "matrix ප්‍රශ්න විසඳන ක්‍රම පෙන්වනවා", "ගණිත ගුරුවරයා", "online class"),
        ("subject", "claim"),
    ),
    HardNegativeCase(
        "result_or_outcome",
        ("Cricket", "ශ්‍රී ලංකා ඉන්දියා තරග ප්‍රතිඵලය", "ශ්‍රී ලංකාව කඩුලු පහකින් ජය ගත්තා", "ක්‍රීඩා වාර්තාව", "කොළඹ"),
        ("Cricket", "ශ්‍රී ලංකා ඉන්දියා තරග ප්‍රතිඵලය", "ඉන්දියාව කඩුලු පහකින් ජය ගත්තා", "ක්‍රීඩා වාර්තාව", "කොළඹ"),
        ("claim",),
    ),
    HardNegativeCase(
        "result_or_outcome",
        ("Government", "සහනාධාර අයදුම්පත් ප්‍රතිඵලය", "අයදුම්පත් අනුමත කරලා", "නිල නිවේදනය", "ශ්‍රී ලංකාව"),
        ("Government", "සහනාධාර අයදුම්පත් ප්‍රතිඵලය", "අයදුම්පත් ප්‍රතික්ෂේප කරලා", "නිල නිවේදනය", "ශ්‍රී ලංකාව"),
        ("claim",),
    ),
    HardNegativeCase(
        "health_or_financial_claim",
        ("Health", "දියවැඩියා ප්‍රතිකාරය", "මෙම පානයෙන් දියවැඩියාව 100% සුව වෙනවා", "health presenter", "online"),
        ("Health", "දියවැඩියා ප්‍රතිකාරය", "මෙම පානය සුවයක් නොවන අතර වෛද්‍ය උපදෙස් අවශ්‍යයි", "health presenter", "online"),
        ("claim",),
    ),
    HardNegativeCase(
        "health_or_financial_claim",
        ("Finance", "crypto ආයෝජනය", "මාසයකින් 20% ලාභය සහතිකයි", "මුදල් උපදේශකයා", "online"),
        ("Finance", "crypto ආයෝජනය", "ලාභය සහතික නැති අතර මුදල් අහිමි වීමේ අවදානම තියෙනවා", "මුදල් උපදේශකයා", "online"),
        ("claim",),
    ),
    HardNegativeCase(
        "time_period",
        ("Jobs", "රජයේ job අයදුම්පත්", "අවසන් දිනය මේ සිකුරාදා", "නිල නිවේදනය", "ශ්‍රී ලංකාව"),
        ("Jobs", "රජයේ job අයදුම්පත්", "අවසන් දිනය ලබන මාසයේ සිකුරාදා", "නිල නිවේදනය", "ශ්‍රී ලංකාව"),
        ("claim",),
    ),
    HardNegativeCase(
        "time_period",
        ("Religion", "පොහොය ධර්ම දේශනාව", "අද සවස 6ට ආරම්භ වෙනවා", "විහාරස්ථානය", "ගම පන්සල"),
        ("Religion", "පොහොය ධර්ම දේශනාව", "හෙට රාත්‍රී 8ට ආරම්භ වෙනවා", "විහාරස්ථානය", "ගම පන්සල"),
        ("claim",),
    ),
]

TOPIC_SCENARIOS = [
    ("Politics", "පාර්ලිමේන්තු විවාදය", "බදාදා විවාදය පැවැත්වෙනවා", "සභා ලේකම් කාර්යාලය", "පාර්ලිමේන්තුව"),
    ("Sri Lankan news", "ඉන්ධන මිල", "ඉදිරි මිල සංශෝධනය ගැන සාකච්ඡා වෙනවා", "CPC නිලධාරීන්", "කොළඹ"),
    ("Economics", "ආර්ථික වර්ධනය", "අලුත් සංඛ්‍යා ලේඛන ගැන පැහැදිලි කිරීමක්", "මහ බැංකු වාර්තාව", "ශ්‍රී ලංකාව"),
    ("Business", "කුඩා ව්‍යාපාර", "online විකුණුම් වැඩි කරගන්න ක්‍රම", "ව්‍යාපාර උපදේශකයා", "කුරුණෑගල"),
    ("Finance", "රන් මිල", "අද වෙළඳපොළේ මිල වෙනස්වීමක් තියෙනවා", "වෙළඳපොළ වාර්තාව", "කොළඹ"),
    ("Health", "උණ රෝගය", "ලක්ෂණ වැඩි නම් වෛද්‍ය උපදෙස් ගන්න", "වෛද්‍යවරයා", "ගම්පහ"),
    ("Technology", "mobile app privacy", "settings වෙනස් කරලා data sharing අඩු කරන්න පුළුවන්", "technology reviewer", "online"),
    ("AI", "AI tool එක", "සිංහලෙන් වැඩ කරන අලුත් features බලනවා", "software trainer", "online"),
    ("Education", "පාසල් නිවාඩු", "සඳුදා පාසල් සාමාන්‍ය පරිදි පැවැත්වෙනවා", "අධ්‍යාපන අමාත්‍යාංශය", "කොළඹ"),
    ("Science", "චන්ද්‍ර ග්‍රහණය", "රාත්‍රී අහසේ දකින්න පුළුවන් අවස්ථාව", "විද්‍යා ගුරුවරයා", "ශ්‍රී ලංකාව"),
    ("Agriculture", "පොහොර බෙදාහැරීම", "ලබන සතියේ ගොවීන්ට පොහොර ලැබෙනවා", "කෘෂිකර්ම නිලධාරියා", "අනුරාධපුර"),
    ("Weather", "වැසි තත්ත්වය", "සවස ගිගුරුම් සහිත වැසි ඇතිවිය හැකියි", "කාලගුණ අංශය", "බස්නාහිර"),
    ("Travel", "ඇල්ල සංචාරය", "අඩු වියදමෙන් දවසක ගමනක් සැලසුම් කරනවා", "travel vlogger", "ඇල්ල"),
    ("Food", "කිරි හොදි recipe", "සරල විදිහට රසට හදන ක්‍රමය", "cooking channel", "නිවස"),
    ("Entertainment", "concert ticket", "tickets online ක්‍රමයට විකිණෙනවා", "සංවිධායක කමිටුව", "කොළඹ"),
    ("Celebrity news", "ජනප්‍රිය නළුවා", "නව චිත්‍රපටය ගැන interview එකක්", "කලා වැඩසටහන", "කොළඹ"),
    ("Sports", "ක්‍රීඩා පුහුණුව", "තරුණ ක්‍රීඩකයන්ට fitness tips දෙන්නේ", "coach", "ක්‍රීඩාංගණය"),
    ("Cricket", "ක්‍රිකට් තරගය", "තරගය හෙට සවස ආරම්භ වෙනවා", "ක්‍රීඩා වාර්තාව", "කොළඹ ක්‍රීඩාංගණය"),
    ("Crime", "පොලිස් පරීක්ෂණය", "සැකකරුවන් දෙදෙනෙක් අත්අඩංගුවට ගත්තා", "පොලිස් මාධ්‍ය ඒකකය", "කුරුණෑගල"),
    ("Religion", "පොහොය වැඩසටහන", "ධර්ම දේශනාව සවස 6ට ආරම්භ වෙනවා", "විහාරස්ථානය", "ගම පන්සල"),
    ("History", "අනුරාධපුර ඉතිහාසය", "පැරණි රාජධානියේ වැදගත් ස්ථාන ගැන කියනවා", "ඉතිහාස ගුරුවරයා", "අනුරාධපුර"),
    ("Social media", "Facebook account security", "password සහ two factor ආරක්ෂාව ගැන බලනවා", "social media creator", "online"),
    ("Government", "passport සේවාව", "online booking කලින් ගන්න ඕන", "ආගමන විගමන දෙපාර්තමේන්තුව", "බත්තරමුල්ල"),
    ("Vehicles", "used car price", "ලංකාවේ second hand වාහන මිල ගැන බලනවා", "vehicle reviewer", "කොළඹ"),
    ("Mobile phones", "අලුත් Android phone", "camera battery performance ගැන review කරනවා", "tech channel", "online"),
    ("Jobs", "රජයේ job අයදුම්පත්", "අවසන් දිනය සිකුරාදා", "නිල නිවේදනය", "ශ්‍රී ලංකාව"),
    ("University", "university cut-off", "අලුත් ලකුණු ගැන සිසුන්ට පැහැදිලි කරනවා", "විශ්වවිද්‍යාල උපදේශකයා", "කැම්පස්"),
    ("Lifestyle", "උදේ routine එක", "වැඩ දවසක් හොඳට පටන් ගන්න ක්‍රම", "lifestyle creator", "නිවස"),
]

OPENERS = [
    "ආයුබෝවන් යාලුවනේ, අද video එකේ අපි {place} පැත්තේ {subject} ගැන කතා කරනවා.",
    "අද episode එක ටිකක් වෙනස්, මොකද ගොඩක් අය comment කරලා අහපු {subject} ගැන මෙහි බලනවා.",
    "මේ update එකේ මුලින්ම කියන්න ඕන දෙය තමයි {subject} ගැන පැතිරෙන ප්‍රශ්නවලට සරල උත්තර දෙන එක.",
    "අද කතාබහ සම්පූර්ණයෙන්ම {subject} වටා යනවා, ඒ නිසා title එකෙන් බලාපොරොත්තු වෙන දේ මෙහි විස්තර වෙනවා.",
]

DETAILS = [
    "{source} කියන විදිහට {claim}.",
    "මේක ගැන social media වලත් කතා වෙන නිසා අපි කරුණු ටික වෙන් කරලා බලමු.",
    "පළවෙනි කරුණ තමයි දිනය, ස්ථානය, සහ අදාළ නිවේදනය වෙන වෙනම තේරුම් ගන්න එක.",
    "කථිකයා viewers ලාට කියන්නේ headline එකට පමණක් නොව සම්පූර්ණ විස්තරය අහලා තීරණ ගන්න කියලා.",
    "මෙහිදී official update, practical effect, සහ ජනතාවට බලපාන පැත්ත ගැන වෙන වෙනම කතා කරනවා.",
    "අවසානයේ short summary එකක් දීලා වැදගත් points තුනක් මතක් කරනවා.",
]

BRIDGES = [
    "හරි, දැන් දෙවෙනි කොටසට යමු.",
    "මේක තේරුම් ගන්න පොඩි example එකක් ගමු.",
    "ඔයාලා දන්නවා ඇති, headline එක සමහර වෙලාවට කෙටි නිසා context එක වැදගත්.",
    "ඒ අතරේ තව පොඩි update එකක්ද තියෙනවා.",
    "මට ලැබුණු තොරතුරු අනුව මෙහි වැදගත්ම point එක මෙන්න.",
]

LANGUAGE_FILLERS = [
    "official", "update", "live", "result", "review", "breaking", "news", "online", "source", "confirmed",
]

MISMATCH_ALT_CLAIMS = [
    "මෙය තවම official ලෙස තහවුරු කරලා නැහැ",
    "දිනය වෙනස් විය හැකි බව පමණක් සඳහන් වෙනවා",
    "rumour එකක් ලෙස මේ ගැන කතා වෙනවා",
    "මෙහි අවසන් තීරණයක් ගැන කියන්නේ නැහැ",
    "පැරණි update එකක් නැවත share වෙනවා",
]


def validate_hard_negative_cases() -> None:
    """Fail fast if a curated pair changes category or undeclared facts."""

    covered_types = {case.change_type for case in HARD_NEGATIVE_CASES}
    if covered_types != HARD_NEGATIVE_CHANGE_TYPES:
        missing = sorted(HARD_NEGATIVE_CHANGE_TYPES - covered_types)
        unexpected = sorted(covered_types - HARD_NEGATIVE_CHANGE_TYPES)
        raise ValueError(f"Invalid hard-negative type coverage. Missing={missing}, unexpected={unexpected}")

    for case in HARD_NEGATIVE_CASES:
        actual_changes = tuple(
            field
            for field, metadata_value, transcript_value in zip(
                SCENARIO_FIELDS,
                case.metadata,
                case.transcript,
                strict=True,
            )
            if metadata_value != transcript_value
        )
        if "topic" in actual_changes:
            raise ValueError(f"Hard negative {case.change_type!r} changes category: {case}")
        if set(actual_changes) != set(case.changed_fields):
            raise ValueError(
                f"Hard negative {case.change_type!r} declares {case.changed_fields}, "
                f"but actually changes {actual_changes}."
            )


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9\u0D80-\u0DFF]+", clean_text(text).lower())


def build_transcript(
    scenario: tuple[str, str, str, str, str],
    row_index: int,
    mode: str,
    target_words: int,
) -> str:
    topic, subject, claim, source, place = scenario
    rng = random.Random(row_index * 7919)
    lines: list[str] = []
    opener = rng.choice(OPENERS).format(subject=subject, place=place)
    lines.append(opener)

    if mode == "match":
        # Always state the decisive fact. Hard negatives reuse this path with one
        # curated fact changed, so the mismatch cannot depend on missing evidence.
        claim_line = DETAILS[0].format(subject=subject, claim=claim, source=source, place=place)
        lines.append(claim_line)
        lines.append(f"මෙම තොරතුර අදාළ වෙන්නේ {place} ප්‍රදේශයටයි.")
        lines.append(f"{subject} ගැන ප්‍රධාන අදහස මේ video එකේ දිගටම explain කරනවා.")
    elif mode == "hard":
        alt = rng.choice(MISMATCH_ALT_CLAIMS)
        lines.append(f"{subject} ගැන කතා කරනවා, හැබැයි {alt}.")
        lines.append(f"{source} ගැන නම් කිහිප තැනක සඳහන් වෙනවා, නමුත් title එකේ පොරොන්දු වෙන අවසන් තීරණය මෙහි කියන්නේ නැහැ.")
    elif mode == "medium":
        lines.append(f"{topic} ක්ෂේත්‍රයේ වෙනත් කරුණක් ගැන මෙහි කතා කරනවා.")
        lines.append(f"{subject} වචනය කිහිප වරක් ඇහුණත් ප්‍රධාන කතාව වෙනත් පැත්තකට යනවා.")
    else:
        lines.append("අද මම කතා කරන්නේ ගෙදර වැඩ, දවසේ අත්දැකීම්, සහ viewers ලාගෙන් ආපු comments ගැන.")
        lines.append("ඉතින් මුලින්ම අපි අද උදේ කරපු වැඩ ටික, කෑම, සහ පොඩි travel plan එක ගැන බලමු.")

    while len(words(" ".join(lines))) < target_words:
        template = rng.choice(DETAILS + BRIDGES)
        if "{claim}" in template:
            if mode == "match":
                text = template.format(subject=subject, claim=claim, source=source, place=place)
            elif mode == "hard":
                text = template.format(subject=subject, claim=rng.choice(MISMATCH_ALT_CLAIMS), source=source, place=place)
            else:
                other = rng.choice(TOPIC_SCENARIOS)
                text = template.format(subject=other[1], claim=other[2], source=other[3], place=other[4])
        else:
            text = template
        if rng.random() < 0.35:
            text += f" {rng.choice(LANGUAGE_FILLERS)} එකත් මෙතැන වැදගත්."
        if rng.random() < 0.25:
            text += f" ref {row_index:05d}-{len(lines):02d}."
        lines.append(text)

    return clean_text(" ".join(lines))


def make_title(scenario: tuple[str, str, str, str, str], row_index: int, label: int, mode: str) -> str:
    _topic, subject, claim, source, place = scenario
    if mode in {"hard", "hard_positive"}:
        return f"{place} {subject}: {source} තහවුරු කළේ {claim} #{row_index:05d}"
    variants = [
        f"{subject} ගැන අලුත්ම update එක #{row_index:05d}",
        f"{place} {subject}: {claim} #{row_index:05d}",
        f"{subject} ගැන දැනගන්න ඕන ප්‍රධාන කරුණු #{row_index:05d}",
        f"{subject} official news සහ සම්පූර්ණ විස්තර #{row_index:05d}",
        f"Breaking update: {subject} ගැන අද තත්ත්වය #{row_index:05d}",
    ]
    return variants[row_index % len(variants)]


def make_description(scenario: tuple[str, str, str, str, str], row_index: int, label: int, mode: str) -> str:
    _topic, subject, claim, source, place = scenario
    if mode in {"hard", "hard_positive"}:
        return f"{place} පැවැත්වෙන {subject} ගැන {source} තහවුරු කළ ප්‍රධාන කරුණ: {claim}."
    variants = [
        f"{place} ප්‍රදේශයේ {subject} ගැන {source} සඳහන් කළ කරුණු සරලව බලමු.",
        f"{subject} සම්බන්ධ latest update, public reaction, සහ practical impact ගැන කෙටි පැහැදිලි කිරීමක්.",
        f"{claim} කියන headline එකට අදාළ background details මෙහි සාකච්ඡා කරනවා.",
        f"Sinhala explanation එකක් ලෙස {subject} ගැන viewers ලා අහපු ප්‍රශ්න කිහිපයකට උත්තර.",
    ]
    if label == 1 and mode in {"medium", "hard"}:
        variants.append(f"{subject} ගැන title එකේ කියන claim එක සම්පූර්ණයෙන් පැහැදිලි කරන video එකක්.")
    return variants[(row_index * 3) % len(variants)]


def make_row(
    row_index: int,
    label: int,
    mode: str,
    hard_case_index: int = 0,
    scenario_index: int | None = None,
    group_id: str = "",
) -> dict[str, str]:
    scenario_position = row_index if scenario_index is None else scenario_index
    scenario = TOPIC_SCENARIOS[scenario_position % len(TOPIC_SCENARIOS)]
    transcript_scenario = scenario
    transcript_mode = "match" if label == 0 else mode
    hard_negative_type = "none"
    if label == 0 and mode == "hard_positive":
        hard_case = HARD_NEGATIVE_CASES[hard_case_index % len(HARD_NEGATIVE_CASES)]
        scenario = hard_case.metadata
        transcript_scenario = hard_case.metadata
    elif label == 1 and mode == "easy":
        transcript_scenario = TOPIC_SCENARIOS[(scenario_position + 9) % len(TOPIC_SCENARIOS)]
    elif label == 1 and mode == "medium":
        transcript_scenario = TOPIC_SCENARIOS[(scenario_position + 3) % len(TOPIC_SCENARIOS)]
    elif label == 1 and mode == "hard":
        hard_case = HARD_NEGATIVE_CASES[hard_case_index % len(HARD_NEGATIVE_CASES)]
        scenario = hard_case.metadata
        transcript_scenario = hard_case.transcript
        transcript_mode = "match"
        hard_negative_type = hard_case.change_type

    target_words = 80 + ((row_index * 37) % 421)
    title = make_title(scenario, row_index, label, mode)
    description = make_description(scenario, row_index, label, mode)
    transcript = build_transcript(transcript_scenario, row_index, transcript_mode, target_words)

    return {
        "video_id": f"SYN{row_index:06d}",
        "group_id": group_id or f"content-pair-{row_index:06d}",
        "title": title,
        "description": description,
        "transcript": transcript,
        "label": str(label),
        "topic": scenario[0],
        "hard_negative_type": hard_negative_type,
    }


def generate_dataset(row_count: int, seed: int) -> tuple[list[dict[str, str]], list[str]]:
    if row_count % 2:
        raise ValueError("row_count must be even so classes can stay balanced.")
    validate_hard_negative_cases()
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    problems: list[str] = []
    per_class = row_count // 2
    mismatch_modes = ["easy", "medium", "hard"]

    for offset in range(1, per_class + 1):
        mode = mismatch_modes[offset % len(mismatch_modes)]
        hard_case_index = (offset - 1) // 3
        group_id = f"content-pair-{offset:06d}"
        positive_mode = "hard_positive" if mode == "hard" else "match"
        rows.append(
            make_row(
                offset,
                0,
                positive_mode,
                hard_case_index=hard_case_index,
                scenario_index=offset,
                group_id=group_id,
            )
        )
        rows.append(
            make_row(
                per_class + offset,
                1,
                mode,
                hard_case_index=hard_case_index,
                scenario_index=offset,
                group_id=group_id,
            )
        )

    rng.shuffle(rows)
    seen_titles: set[str] = set()
    seen_transcripts: set[str] = set()
    valid_rows: list[dict[str, str]] = []
    for row in rows:
        if row["title"] in seen_titles:
            problems.append(f"Duplicate title skipped: {row['video_id']}")
            continue
        if row["transcript"] in seen_transcripts:
            problems.append(f"Duplicate transcript skipped: {row['video_id']}")
            continue
        seen_titles.add(row["title"])
        seen_transcripts.add(row["transcript"])
        valid_rows.append(row)
    if len(valid_rows) != row_count:
        problems.append(f"Expected {row_count} rows, generated {len(valid_rows)} after duplicate filtering.")
    return valid_rows, problems


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def cosine(left: Counter[str], right: Counter[str]) -> float:
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def near_duplicate_count(rows: list[dict[str, str]], threshold: float = 0.96, sample_limit: int = 1200) -> int:
    sampled = rows[:sample_limit]
    vectors = [Counter(words(row["transcript"])) for row in sampled]
    count = 0
    for i, vector in enumerate(vectors):
        for j in range(i + 1, len(vectors)):
            if cosine(vector, vectors[j]) >= threshold:
                count += 1
    return count


def row_topic(row: dict[str, str]) -> str:
    explicit_topic = clean_text(row.get("topic", ""))
    if explicit_topic:
        return explicit_topic
    text = f"{row['title']} {row['description']}"
    best_topic = "Unknown"
    for topic, subject, *_rest in TOPIC_SCENARIOS:
        if subject in text:
            best_topic = topic
            break
    return best_topic


def validate_dataset(rows: list[dict[str, str]], problems: list[str]) -> str:
    labels = Counter(row["label"] for row in rows)
    titles = Counter(row["title"] for row in rows)
    transcripts = Counter(row["transcript"] for row in rows)
    missing = {
        column: sum(1 for row in rows if not clean_text(row.get(column, "")))
        for column in OUTPUT_COLUMNS
    }
    lengths_by_label: dict[str, list[int]] = defaultdict(list)
    topics_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    hard_negative_types: Counter[str] = Counter()
    for row in rows:
        lengths_by_label[row["label"]].append(len(words(row["transcript"])))
        topics_by_label[row["label"]][row_topic(row)] += 1
        hard_negative_types[row["hard_negative_type"]] += 1

    duplicate_rows = len(rows) - len({tuple(row[column] for column in OUTPUT_COLUMNS) for row in rows})
    duplicate_titles = sum(count - 1 for count in titles.values() if count > 1)
    duplicate_transcripts = sum(count - 1 for count in transcripts.values() if count > 1)
    near_duplicates = near_duplicate_count(rows)

    leakage_terms = ["mismatch", "matching", "unrelated content", "label", "false"]
    leakage_hits = {
        term: sum(1 for row in rows if term in row["transcript"].lower())
        for term in leakage_terms
    }

    lines = [
        "Sinhala YouTube Content Match Dataset Quality Report",
        "",
        f"total samples: {len(rows)}",
        f"MATCH label 0 samples: {labels.get('0', 0)}",
        f"MISMATCH label 1 samples: {labels.get('1', 0)}",
        f"exact duplicate rows: {duplicate_rows}",
        f"duplicate titles: {duplicate_titles}",
        f"duplicate transcripts: {duplicate_transcripts}",
        f"near duplicate transcript pairs in first 1200 rows at cosine >= 0.96: {near_duplicates}",
        "",
        "missing values:",
        *[f"- {column}: {count}" for column, count in missing.items()],
        "",
        "average transcript word length per class:",
        *[
            f"- label {label}: {sum(lengths) / len(lengths):.1f}"
            for label, lengths in sorted(lengths_by_label.items())
            if lengths
        ],
        "",
        "topic distribution by label:",
    ]
    for label in sorted(topics_by_label):
        lines.append(f"- label {label}: {dict(sorted(topics_by_label[label].items()))}")
    lines.extend(
        [
            "",
            "hard-negative fact changes:",
            *[f"- {change_type}: {count}" for change_type, count in sorted(hard_negative_types.items())],
            "",
            "label leakage term hits in transcript:",
            *[f"- {term}: {count}" for term, count in leakage_hits.items()],
            "",
            "generation problems found:",
            *(f"- {problem}" for problem in problems),
        ]
    )
    if not problems:
        lines.append("- none")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", default="data/sinhala_youtube_content_match_dataset.csv")
    parser.add_argument("--report", default="data/dataset_quality_report.txt")
    args = parser.parse_args()

    rows, problems = generate_dataset(args.rows, args.seed)
    write_csv(rows, Path(args.output))
    report = validate_dataset(rows, problems)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote quality report to {args.report}")
    print(report)


if __name__ == "__main__":
    main()

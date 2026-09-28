"""Build a human-friendly Sinhala synthetic pool for similarity/alignment labeling."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.consistency import compute_consistency_features
from utils.data_utils import clean_text


OUTPUT_COLUMNS = [
    "synthetic_id",
    "video_id",
    "url",
    "title",
    "description",
    "transcript",
    "topic",
    "seed_binary_label",
    "seed_consistency_label",
    "seed_mismatch_type",
    "seed_support_rationale",
    "generated_negative",
    "source_title_video_id",
    "source_transcript_video_id",
    "title_transcript_similarity",
    "description_transcript_similarity",
    "title_description_similarity",
    "similarity_band",
    "suggested_group_id",
    "label_status",
    "reviewed_binary_label",
    "reviewed_consistency_label",
    "reviewed_mismatch_type",
    "reviewed_support_rationale",
    "reviewed_at",
]

SCENARIOS = [
    {
        "topic": "Education",
        "subject": "පාසල් නිවාඩු",
        "claim": "සඳුදා පාසල් නිවාඩු නැහැ",
        "alt_claim": "සඳුදා පාසල් නිවාඩු දීලා තියෙනවා",
        "place": "කොළඹ",
        "speaker": "අධ්‍යාපන අමාත්‍යාංශය",
        "detail": "සාමාන්‍ය පරිදි පාසල් පැවැත්වෙනවා",
    },
    {
        "topic": "Health",
        "subject": "උණ රෝගය",
        "claim": "ළඟම රෝහලෙන් වෛද්‍ය උපදෙස් ගන්න",
        "alt_claim": "ගෙදර බෙහෙත් වලින් 100% සුව වෙනවා",
        "place": "ගම්පහ",
        "speaker": "වෛද්‍යවරයා",
        "detail": "උණ වැඩි නම් doctor කෙනෙක් හමුවෙන්න",
    },
    {
        "topic": "Finance",
        "subject": "රන් මිල",
        "claim": "අද රන් මිල ටිකක් ඉහළ ගියා",
        "alt_claim": "අද රන් මිල විශාල ලෙස පහළ ගියා",
        "place": "කොළඹ වෙළඳපොළ",
        "speaker": "වෙළඳපොළ වාර්තාව",
        "detail": "මිල වෙනස්වීම සුළුයි කියලා පැහැදිලි කරනවා",
    },
    {
        "topic": "Weather",
        "subject": "වැසි තත්ත්වය",
        "claim": "සවස වැසි ඇතිවිය හැකියි",
        "alt_claim": "අද කිසිම වැස්සක් නැහැ",
        "place": "බස්නාහිර පළාත",
        "speaker": "කාලගුණ අංශය",
        "detail": "සවස ගිගුරුම් සහිත වැසි ඇති වෙන්න පුළුවන්",
    },
    {
        "topic": "Government",
        "subject": "passport සේවාව",
        "claim": "online booking කලින් ගන්න ඕන",
        "alt_claim": "booking නැතුව ඕනෑම වෙලාවක යන්න පුළුවන්",
        "place": "බත්තරමුල්ල",
        "speaker": "ආගමන විගමන දෙපාර්තමේන්තුව",
        "detail": "සේවාවට පෙර online appointment එකක් අවශ්‍යයි",
    },
    {
        "topic": "Jobs",
        "subject": "රජයේ job අයදුම්පත්",
        "claim": "අවසන් දිනය සිකුරාදා",
        "alt_claim": "අවසන් දිනය අද රෑ 12ට",
        "place": "ශ්‍රී ලංකාව",
        "speaker": "නිල නිවේදනය",
        "detail": "අයදුම්පත් සිකුරාදා දක්වා භාරගන්නවා",
    },
    {
        "topic": "Technology",
        "subject": "mobile app privacy",
        "claim": "settings වෙනස් කරලා data sharing අඩු කරන්න පුළුවන්",
        "alt_claim": "app එක සියලු data හොරෙන් විකුණනවා",
        "place": "online",
        "speaker": "technology reviewer",
        "detail": "privacy settings පරීක්ෂා කරන ආකාරය පෙන්වනවා",
    },
    {
        "topic": "Transport",
        "subject": "බස් ගාස්තු",
        "claim": "නව ගාස්තුව ලබන සතියේ සිටයි",
        "alt_claim": "නව ගාස්තුව අද සිටම ක්‍රියාත්මකයි",
        "place": "කොළඹ",
        "speaker": "ප්‍රවාහන අධිකාරිය",
        "detail": "ගාස්තු වෙනස ලබන සතියේ සිට ක්‍රියාත්මක වෙනවා",
    },
    {
        "topic": "Energy",
        "subject": "විදුලි කප්පාදුව",
        "claim": "අද රාත්‍රියේ පැයක කප්පාදුවක් තියෙනවා",
        "alt_claim": "අද මුළු දවසම විදුලිය නැහැ",
        "place": "මාතර",
        "speaker": "විදුලිබල මණ්ඩලය",
        "detail": "රාත්‍රී කාලයේ පැයක පමණ කප්පාදුවක් කියලා සඳහන් වෙනවා",
    },
    {
        "topic": "Sports",
        "subject": "ක්‍රිකට් තරගය",
        "claim": "තරගය හෙට සවස ආරම්භ වෙනවා",
        "alt_claim": "තරගය අද උදේ අවසන් වුණා",
        "place": "කොළඹ ක්‍රීඩාංගණය",
        "speaker": "ක්‍රීඩා වාර්තාව",
        "detail": "තරග කාලසටහන හෙට සවස කියලා පැහැදිලි කරනවා",
    },
    {
        "topic": "Crime",
        "subject": "පොලිස් පරීක්ෂණය",
        "claim": "සැකකරුවන් දෙදෙනෙක් අත්අඩංගුවට ගත්තා",
        "alt_claim": "සියලු සැකකරුවන් නිදහස් කළා",
        "place": "කුරුණෑගල",
        "speaker": "පොලිස් මාධ්‍ය ඒකකය",
        "detail": "දෙදෙනෙක් අත්අඩංගුවට ගත් බව කියනවා",
    },
    {
        "topic": "Agriculture",
        "subject": "පොහොර බෙදාහැරීම",
        "claim": "ගොවීන්ට ලබන සතියේ පොහොර ලැබෙනවා",
        "alt_claim": "පොහොර බෙදාහැරීම සම්පූර්ණයෙන් නවතා දැම්මා",
        "place": "අනුරාධපුර",
        "speaker": "කෘෂිකර්ම නිලධාරියා",
        "detail": "ලබන සතියේ බෙදාහැරීම ආරම්භ වෙනවා",
    },
    {
        "topic": "Politics",
        "subject": "පාර්ලිමේන්තු විවාදය",
        "claim": "විවාදය ලබන බදාදා පැවැත්වෙනවා",
        "alt_claim": "විවාදය සම්පූර්ණයෙන් අවලංගුයි",
        "place": "පාර්ලිමේන්තුව",
        "speaker": "සභා ලේකම් කාර්යාලය",
        "detail": "බදාදා විවාදය පැවැත්වීමට කාලය වෙන් කරලා තියෙනවා",
    },
    {
        "topic": "Religion",
        "subject": "පොහොය වැඩසටහන",
        "claim": "ධර්ම දේශනාව සවස 6ට ආරම්භ වෙනවා",
        "alt_claim": "වැඩසටහන සම්පූර්ණයෙන් අවලංගු කරලා",
        "place": "ගම පන්සල",
        "speaker": "විහාරස්ථානය",
        "detail": "සවස 6ට දේශනාව ආරම්භ කරන බව දැනුම් දෙනවා",
    },
    {
        "topic": "Entertainment",
        "subject": "concert ticket",
        "claim": "tickets online විතරයි ගන්න පුළුවන්",
        "alt_claim": "tickets නොමිලේ දෙනවා",
        "place": "නෙළුම් පොකුණ",
        "speaker": "සංවිධායක කමිටුව",
        "detail": "ප්‍රවේශපත්‍ර online ක්‍රමයට විකිණෙනවා",
    },
]

DIFFERENT_TRANSCRIPTS = [
    "මේ video එකේ කතා කරන්නේ ගෙවත්තේ මල් පැළ වලට වතුර දාන්නේ කොහොමද කියලා. title එකේ කියන දේ මෙහි සාකච්ඡා වෙන්නේ නැහැ.",
    "මෙහි ප්‍රධාන වශයෙන් පෙන්වන්නේ කුඩා කෑම වට්ටෝරුවක්. මිල, නිවේදන, හෝ official update ගැන කිසිම පැහැදිලි කිරීමක් නැහැ.",
    "කථිකයා අද කතා කරන්නේ පවුලේ ගමනක් සහ දවසේ අත්දැකීම් ගැන. title එකේ claim එකට transcript එකෙන් support එකක් නැහැ.",
    "මේ transcript එකේ තියෙන්නේ පරණ ගීතයක් ගැන සාකච්ඡාවක්. description එකේ පොරොන්දු වෙන තොරතුරු මෙහි නැහැ.",
]


def sentence_pack(scenario: dict[str, str], extra: str = "") -> str:
    return (
        f"අද video එකේ {scenario['place']} ප්‍රදේශයේ {scenario['subject']} ගැන කතා කරනවා. "
        f"{scenario['speaker']} කියන විදිහට {scenario['claim']}. "
        f"{scenario['detail']}. "
        f"ඒ නිසා title සහ description දෙකම transcript එකෙන් support වෙනවා. {extra}"
    ).strip()


def make_title(scenario: dict[str, str], style: str, index: int) -> str:
    if style == "direct":
        return f"{scenario['subject']} update: {scenario['claim']}"
    if style == "natural":
        return f"{scenario['subject']} ගැන අද දැනගන්න ඕන දේ"
    if style == "short":
        return f"{scenario['subject']} ගැන අලුත්ම විස්තර"
    if style == "overclaim":
        return f"Breaking! {scenario['subject']} ගැන 100% තහවුරු වූ රහස"
    if style == "wrong_detail":
        return f"{scenario['subject']} update: {scenario['alt_claim']}"
    return f"{scenario['subject']} review #{index}"


def make_description(scenario: dict[str, str], style: str) -> str:
    if style == "direct":
        return f"{scenario['speaker']} ලබාදුන් update එක අනුව {scenario['claim']} කියලා මෙහි පැහැදිලි කරනවා."
    if style == "natural":
        return f"{scenario['subject']} ගැන සරල සිංහල පැහැදිලි කිරීමක්. ප්‍රධාන කරුණු transcript එකේ තියෙනවා."
    if style == "wrong_detail":
        return f"මෙම video එකේ {scenario['alt_claim']} කියන official තොරතුර සම්පූර්ණයෙන් දෙනවා."
    if style == "overclaim":
        return f"කවුරුත් නොකියන shocking news එකක්. සියලු නම්, දිනයන් සහ රහස් මෙහි තියෙනවා."
    return f"{scenario['subject']} ගැන කෙටි update එකක්."


def similarity_band(label: str, kind: str) -> str:
    if label == "Not False":
        return {
            "direct": "high_similarity_consistent",
            "natural": "medium_similarity_consistent",
            "short": "low_similarity_consistent",
        }[kind]
    return {
        "different": "low_similarity_false",
        "wrong_detail": "high_similarity_false",
        "overclaim": "medium_similarity_false",
        "weak": "medium_similarity_false",
    }[kind]


def base_row(
    row_number: int,
    scenario: dict[str, str],
    title: str,
    description: str,
    transcript: str,
    label: str,
    mismatch_type: str,
    rationale: str,
    band: str,
    source_transcript_id: str = "",
) -> dict[str, str]:
    video_id = f"human-sim-{row_number:05d}"
    source_title_id = f"source-title-{row_number:05d}"
    day = (row_number % 28) + 1
    month_note = f"අගෝස්තු {day} update."
    row = {
        "synthetic_id": f"syn-human-{row_number:05d}",
        "video_id": video_id,
        "url": f"https://synthetic.local/watch?v={video_id}",
        "title": title,
        "description": f"{description} {month_note}",
        "transcript": f"{transcript} review ref {row_number:05d}.",
        "topic": scenario["topic"],
        "seed_binary_label": label,
        "seed_consistency_label": "Consistent" if label == "Not False" else "Inconsistent",
        "seed_mismatch_type": mismatch_type,
        "seed_support_rationale": rationale,
        "generated_negative": "false" if label == "Not False" else "true",
        "source_title_video_id": source_title_id,
        "source_transcript_video_id": source_transcript_id or source_title_id,
        "similarity_band": band,
        "suggested_group_id": f"human-friendly-{row_number:05d}",
        "label_status": "new",
        "reviewed_binary_label": "",
        "reviewed_consistency_label": "",
        "reviewed_mismatch_type": "",
        "reviewed_support_rationale": "",
        "reviewed_at": "",
    }
    features = compute_consistency_features(title, description, transcript).as_dict()
    for key, value in features.items():
        row[key] = f"{value:.4f}"
    return row


def consistent_rows(row_number: int, scenario: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for offset, style in enumerate(["direct", "natural", "short"]):
        rows.append(
            base_row(
                row_number + offset,
                scenario,
                make_title(scenario, style, row_number + offset),
                make_description(scenario, style),
                sentence_pack(scenario),
                "Not False",
                "consistent",
                "Transcript supports the main title and description claim.",
                similarity_band("Not False", style),
            )
        )
    return rows


def false_rows(row_number: int, scenario: dict[str, str], rng: random.Random, cycle: int) -> list[dict[str, str]]:
    different_transcript = rng.choice(DIFFERENT_TRANSCRIPTS)
    different_id = f"other-transcript-{row_number:05d}"
    weak_transcript = (
        f"මෙහි {scenario['subject']} ගැන සාමාන්‍යයෙන් කතා කරනවා. "
        "නමුත් title එකේ තියෙන නිශ්චිත claim එක, දිනය, හෝ official තීරණය transcript එකේ තහවුරු කරන්නේ නැහැ."
    )
    medium_false = (
        base_row(
            row_number + 2,
            scenario,
            make_title(scenario, "overclaim", row_number + 2),
            make_description(scenario, "overclaim"),
            sentence_pack(scenario, "රහස් ලැයිස්තු හෝ shocking තොරතුරු transcript එකේ නැහැ."),
            "False",
            "clickbait_overclaim",
            "Title and description overclaim compared with the transcript.",
            similarity_band("False", "overclaim"),
            source_transcript_id=f"supporting-{row_number + 2:05d}",
        )
        if cycle % 2 == 0
        else base_row(
            row_number + 2,
            scenario,
            make_title(scenario, "short", row_number + 2),
            make_description(scenario, "wrong_detail"),
            weak_transcript,
            "False",
            "weak_support",
            "Transcript mentions the subject but does not support the promised claim.",
            similarity_band("False", "weak"),
            source_transcript_id=f"weak-{row_number + 2:05d}",
        )
    )
    rows = [
        base_row(
            row_number,
            scenario,
            make_title(scenario, "direct", row_number),
            make_description(scenario, "direct"),
            different_transcript,
            "False",
            "title_transcript_mismatch",
            "Title and description are about one claim, but transcript is about a completely different thing.",
            similarity_band("False", "different"),
            source_transcript_id=different_id,
        ),
        base_row(
            row_number + 1,
            scenario,
            make_title(scenario, "wrong_detail", row_number + 1),
            make_description(scenario, "wrong_detail"),
            sentence_pack(scenario, "මෙහි වෙනස් දිනයක් හෝ වෙනස් ප්‍රතිඵලයක් කියන්නේ නැහැ."),
            "False",
            "title_transcript_mismatch",
            "Metadata repeats the same topic words but changes the actual claim.",
            similarity_band("False", "wrong_detail"),
            source_transcript_id=f"supporting-{row_number + 1:05d}",
        ),
        medium_false,
    ]
    return rows


def generate_rows(row_count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    row_number = 1
    cycle = 0
    while len(rows) < row_count:
        scenarios = list(SCENARIOS)
        rng.shuffle(scenarios)
        for scenario in scenarios:
            rows.extend(consistent_rows(row_number, scenario))
            row_number += 3
            rows.extend(false_rows(row_number, scenario, rng, cycle))
            row_number += 3
            cycle += 1
            if len(rows) >= row_count:
                break
    rows = rows[:row_count]
    rng.shuffle(rows)
    return rows


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean_text(row.get(column, "")) for column in OUTPUT_COLUMNS})


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, int] | int]:
    summary: dict[str, dict[str, int] | int] = {
        "total_rows": len(rows),
        "labels": {},
        "bands": {},
        "mismatch_types": {},
    }
    for row in rows:
        for key, column in [
            ("labels", "seed_binary_label"),
            ("bands", "similarity_band"),
            ("mismatch_types", "seed_mismatch_type"),
        ]:
            bucket = summary[key]
            assert isinstance(bucket, dict)
            value = row[column]
            bucket[value] = bucket.get(value, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/synthetic_similarity_labeling_pool.csv")
    args = parser.parse_args()

    rows = generate_rows(max(args.rows, 105), args.seed)
    output_path = Path(args.output)
    write_rows(rows, output_path)
    print(f"Wrote {len(rows)} human-friendly synthetic rows to {output_path}")
    print(summarize(rows))


if __name__ == "__main__":
    main()

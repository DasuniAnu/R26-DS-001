"""Build a Sinhala consistency labeling pool from deterministic synthetic cases."""

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
    "suggested_group_id",
    "label_status",
    "reviewed_binary_label",
    "reviewed_at",
]

TOPIC_CASES = {
    "Agriculture": [
        ("වී වගාව", "පොහොර සහනාධාරය", "කෘෂිකර්ම නිලධාරීන්ගේ උපදෙස්"),
        ("තේ වගාව", "නව රෝග පාලන ක්‍රම", "ගොවීන්ට ලබාදෙන උපදෙස්"),
        ("එළවළු මිල", "වෙළඳපොළ update", "කොළඹ සහ දඹුල්ල මිල වෙනස්වීම්"),
        ("ජල සැපයුම", "ගොවිබිම් සඳහා කාලසටහන", "ප්‍රදේශ අනුව ලබාදෙන ජල මුර"),
    ],
    "Education": [
        ("O/L exam", "ප්‍රතිඵල නිකුත් කිරීම", "විභාග දෙපාර්තමේන්තුවේ නිවේදනය"),
        ("පාසල් නිවාඩු", "කාලසටහන් වෙනස්වීම", "අධ්‍යාපන අමාත්‍යාංශ update"),
        ("විශ්වවිද්‍යාල ප්‍රවේශය", "නව cut-off ලකුණු", "අයදුම්කරුවන්ට උපදෙස්"),
        ("ශිෂ්‍යත්ව exam", "අයදුම්පත් දිනය", "පාසල් මට්ටමේ දැනුම්දීම්"),
    ],
    "Entertainment": [
        ("නව චිත්‍රපටය", "release දිනය", "අධ්‍යක්ෂවරයාගේ පැහැදිලි කිරීම"),
        ("ජනප්‍රිය ගායකයා", "concert update", "ප්‍රවේශපත්‍ර සහ ස්ථානය"),
        ("ටෙලි නාට්‍යය", "අලුත් episode", "කතාවේ ඉදිරි වෙනස්කම්"),
        ("viral වීඩියෝව", "සමාජ මාධ්‍ය ප්‍රතිචාර", "කලාකරුවන්ගේ අදහස්"),
    ],
    "Finance": [
        ("බැංකු loan", "පොලී අනුපාත update", "අයදුම් කිරීමේ කොන්දේසි"),
        ("රන් මිල", "අද වෙළඳපොළ තත්ත්වය", "මිල වෙනස්වීමට හේතු"),
        ("crypto investment", "අවදානම් සහ වාසි", "මුදල් උපදේශකගේ පැහැදිලි කිරීම"),
        ("විදුලි බිල් සහනය", "ගෙවීම් ක්‍රම", "පාරිභෝගිකයින්ට උපදෙස්"),
    ],
    "Government": [
        ("රජයේ allowance", "official අයදුම් ක්‍රමය", "ලැබිය යුතු පවුල් කාණ්ඩ"),
        ("passport සේවාව", "online booking update", "කාර්යාල වේලාවන්"),
        ("gazette නිවේදනය", "නව නියමයන්", "අදාළ දිනය සහ බලපෑම"),
        ("රජයේ job", "අයදුම්පත් කැඳවීම", "සුදුසුකම් සහ අවසන් දිනය"),
    ],
    "Health": [
        ("vaccine update", "රෝහල් ලබාදෙන දිනය", "වෛද්‍යවරයාගේ උපදෙස්"),
        ("දියවැඩියාව", "ආහාර පාලන ක්‍රම", "doctor පැහැදිලි කිරීම"),
        ("උණ රෝගය", "ලක්ෂණ සහ ප්‍රතිකාර", "සෞඛ්‍ය අංශ අනතුරු ඇඟවීම"),
        ("medicine shortage", "රෝහල් තත්ත්වය", "නිල සැපයුම් update"),
    ],
    "Politics": [
        ("පාර්ලිමේන්තු විවාදය", "අද තීරණය", "මන්ත්‍රීවරුන්ගේ අදහස්"),
        ("election update", "ඡන්ද දිනය ගැන පැහැදිලි කිරීම", "කොමිසමේ නිවේදනය"),
        ("අමාත්‍ය මණ්ඩලය", "නව තීරණ", "මාධ්‍ය හමුවේ විස්තර"),
        ("පක්ෂ රැස්වීම", "ප්‍රතිපත්ති ප්‍රකාශය", "නායකත්වයේ අදහස්"),
    ],
    "Religion": [
        ("පොහොය වැඩසටහන", "පන්සල් කාලසටහන", "දායකයින්ට දැනුම්දීම"),
        ("ධර්ම දේශනාව", "සජීවී වැඩසටහන", "භික්ෂූන් වහන්සේගේ පැහැදිලි කිරීම"),
        ("පුණ්‍ය කටයුතු", "උත්සව සංවිධානය", "ප්‍රදේශවාසීන්ට උපදෙස්"),
        ("ආගමික උත්සවය", "ගමනාගමන සැලැස්ම", "පොලීසියේ දැනුම්දීම"),
    ],
    "Technology": [
        ("mobile app", "privacy update", "දත්ත ආරක්ෂාව ගැන පැහැදිලි කිරීම"),
        ("AI tool", "සිංහල භාවිතය", "නව feature සහ සීමා"),
        ("Facebook account", "security warning", "password ආරක්ෂාව"),
        ("data package", "නව මිල", "telecom company update"),
    ],
    "Weather": [
        ("වැසි තත්ත්වය", "කාලගුණ warning", "දිස්ත්‍රික්ක අනුව අනාවැකි"),
        ("සුළි කුණාටු අවදානම", "official forecast", "මුහුදු ප්‍රදේශ අනතුරු ඇඟවීම"),
        ("ගංවතුර තත්ත්වය", "ජල මට්ටම් update", "ආපදා කළමනාකරණ උපදෙස්"),
        ("උෂ්ණත්වය", "අද දවසේ වෙනස්වීම්", "සෞඛ්‍ය උපදෙස්"),
    ],
}

OPENERS = [
    "අද වීඩියෝවෙන් අපි කතා කරන්නේ {subject} ගැනයි.",
    "මේ update එකේ ප්‍රධාන කරුණ {subject} සම්බන්ධයෙන් ලැබුණු තොරතුරුයි.",
    "ගොඩක් අය අහන ප්‍රශ්නයක් වන {subject} ගැන මෙහි පැහැදිලි කරනවා.",
]

DETAIL_LINES = [
    "{detail} ගැන කථිකයා පියවරෙන් පියවර පැහැදිලි කරයි.",
    "මෙහිදී source ලෙස සඳහන් කරන්නේ {evidence} යන්නයි.",
    "video එක පුරාම ප්‍රධාන අවධානය {detail} මතම තියෙනවා.",
    "අවසානයේ viewers ලාට නිල update බලලා තීරණ ගන්න කියනවා.",
]

CLICKBAIT_TITLES = [
    "Breaking! {subject} ගැන කවුරුත් නොකියන භයානක රහස",
    "දැන්ම බලන්න! {subject} නිසා ලංකාවම කම්පා වෙයි",
    "100% තහවුරුයි! {subject} ගැන official තීරණය එළියට",
    "ඔවුන් මේක සඟවනවා! {subject} ගැන සම්පූර්ණ ඇත්ත",
]

WEAK_TITLES = [
    "{subject} ගැන ලොකු වෙනසක්? සම්පූර්ණ update මෙන්න",
    "{subject} සම්බන්ධ නව තීරණයක්ද? ජනතාව අහන ප්‍රශ්න",
    "{subject} ගැන හදිසි warning එකක්ද? අද තත්ත්වය",
]


def build_transcript(subject: str, detail: str, evidence: str, variant: int) -> str:
    opener = OPENERS[variant % len(OPENERS)].format(subject=subject)
    lines = [
        opener,
        DETAIL_LINES[(variant + 0) % len(DETAIL_LINES)].format(detail=detail, evidence=evidence),
        DETAIL_LINES[(variant + 1) % len(DETAIL_LINES)].format(detail=detail, evidence=evidence),
        "මෙය external fact check එකක් නොව title, description සහ transcript එක එකිනෙකට ගැළපෙනවාද කියලා බලන උදාහරණයක්.",
        "වැදගත්ම දෙය නම් claim එකට transcript එක තුළ ප්‍රමාණවත් support තියෙනවාද කියලා පරීක්ෂා කිරීමයි.",
    ]
    return " ".join(lines)


def with_features(row: dict[str, str]) -> dict[str, str]:
    features = compute_consistency_features(row["title"], row["description"], row["transcript"]).as_dict()
    for column, value in features.items():
        row[column] = f"{value:.3f}"
    return row


def make_consistent_row(topic: str, case: tuple[str, str, str], index: int) -> dict[str, str]:
    subject, detail, evidence = case
    video_id = f"SYN{index:05d}"
    title = f"{subject} ගැන official update - {detail}"
    description = f"{subject} සම්බන්ධ {detail} සහ {evidence} ගැන සිංහල පැහැදිලි කිරීමක්."
    transcript = build_transcript(subject, detail, evidence, index)
    row = {
        "synthetic_id": video_id,
        "video_id": video_id,
        "url": f"https://synthetic.local/watch?v={video_id}",
        "title": title,
        "description": description,
        "transcript": transcript,
        "topic": topic,
        "seed_binary_label": "Not False",
        "seed_consistency_label": "Consistent",
        "seed_mismatch_type": "consistent",
        "seed_support_rationale": "Title, description, and transcript discuss the same subject and supported detail.",
        "generated_negative": "false",
        "source_title_video_id": video_id,
        "source_transcript_video_id": video_id,
        "suggested_group_id": f"synthetic-{topic.lower()}-{video_id.lower()}",
        "label_status": "new",
        "reviewed_binary_label": "",
        "reviewed_at": "",
    }
    return with_features(row)


def make_negative_row(
    source: dict[str, str],
    target: dict[str, str],
    index: int,
    mismatch_type: str,
) -> dict[str, str]:
    video_id = f"SYNNEG{index:05d}"
    row = dict(source)
    row["synthetic_id"] = video_id
    row["video_id"] = video_id
    row["url"] = f"https://synthetic.local/watch?v={video_id}"
    row["seed_binary_label"] = "False"
    row["seed_consistency_label"] = "Inconsistent"
    row["seed_mismatch_type"] = mismatch_type
    row["generated_negative"] = "true"
    row["source_title_video_id"] = source["video_id"]
    row["source_transcript_video_id"] = target["video_id"]
    row["suggested_group_id"] = f"synthetic-{source['topic'].lower()}-{video_id.lower()}"
    row["label_status"] = "new"
    row["reviewed_binary_label"] = ""
    row["reviewed_at"] = ""

    if mismatch_type == "same_topic_swap":
        row["transcript"] = target["transcript"]
        row["seed_support_rationale"] = "Same-topic hard negative: metadata and transcript are from different synthetic videos."
    elif mismatch_type == "title_transcript_mismatch":
        row["title"] = CLICKBAIT_TITLES[index % len(CLICKBAIT_TITLES)].format(
            subject=clean_text(source["title"]).split(" ගැන")[0]
        )
        row["seed_support_rationale"] = "Title makes a stronger promise than the transcript supports."
    elif mismatch_type == "description_transcript_mismatch":
        row["description"] = (
            f"{clean_text(source['title']).split(' ගැන')[0]} ගැන සියලු official ලේඛන, නම් ලැයිස්තු, "
            "දිනයන් සහ රහස් තොරතුරු transcript එකේ සම්පූර්ණයෙන් ඇති බව කියයි."
        )
        row["seed_support_rationale"] = "Description promises detailed evidence that the transcript does not contain."
    elif mismatch_type == "clickbait_overclaim":
        row["title"] = CLICKBAIT_TITLES[(index + 1) % len(CLICKBAIT_TITLES)].format(
            subject=clean_text(source["title"]).split(" ගැන")[0]
        )
        row["description"] = "මෙම video එක 100% තහවුරු වූ breaking news එකක් කියලා overclaim කරයි."
        row["seed_support_rationale"] = "Metadata overclaims certainty while transcript is only a general discussion."
    else:
        row["title"] = WEAK_TITLES[index % len(WEAK_TITLES)].format(subject=clean_text(source["title"]).split(" ගැන")[0])
        row["seed_support_rationale"] = "Transcript mentions the topic but does not support the main promised claim."
    return with_features(row)


def generate_rows(row_count: int = 1000, seed: int = 42) -> list[dict[str, str]]:
    random.seed(seed)
    consistent_rows: list[dict[str, str]] = []
    consistent_target = row_count // 2
    index = 1
    while len(consistent_rows) < consistent_target:
        for case_index in range(max(len(cases) for cases in TOPIC_CASES.values())):
            for topic, cases in TOPIC_CASES.items():
                if len(consistent_rows) >= consistent_target:
                    break
                case = cases[case_index % len(cases)]
                consistent_rows.append(make_consistent_row(topic, case, index))
                index += 1
            if len(consistent_rows) >= consistent_target:
                break

    by_topic: dict[str, list[dict[str, str]]] = {}
    for row in consistent_rows:
        by_topic.setdefault(row["topic"], []).append(row)

    mismatch_types = [
        "same_topic_swap",
        "title_transcript_mismatch",
        "description_transcript_mismatch",
        "clickbait_overclaim",
        "weak_support",
    ]
    negative_rows: list[dict[str, str]] = []
    neg_index = 1
    topics = list(by_topic)
    while len(consistent_rows) + len(negative_rows) < row_count:
        topic = topics[neg_index % len(topics)]
        pool = by_topic[topic]
        source = pool[neg_index % len(pool)]
        target = pool[(neg_index + 1) % len(pool)]
        mismatch_type = mismatch_types[neg_index % len(mismatch_types)]
        negative_rows.append(make_negative_row(source, target, neg_index, mismatch_type))
        neg_index += 1

    rows = consistent_rows + negative_rows
    random.shuffle(rows)
    return rows[:row_count]


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean_text(row.get(column, "")) for column in OUTPUT_COLUMNS})


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, int] | int]:
    summary: dict[str, dict[str, int] | int] = {"total_rows": len(rows), "labels": {}, "topics": {}, "mismatch_types": {}}
    for row in rows:
        for key, column in [
            ("labels", "seed_binary_label"),
            ("topics", "topic"),
            ("mismatch_types", "seed_mismatch_type"),
        ]:
            bucket = summary[key]
            assert isinstance(bucket, dict)
            value = row[column]
            bucket[value] = bucket.get(value, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic_consistency_labeling_pool.csv")
    args = parser.parse_args()

    rows = generate_rows(row_count=max(args.rows, 20), seed=args.seed)
    output_path = Path(args.output)
    write_rows(rows, output_path)
    print(f"Wrote {len(rows)} synthetic labeling rows to {output_path}")
    print(summarize(rows))


if __name__ == "__main__":
    main()

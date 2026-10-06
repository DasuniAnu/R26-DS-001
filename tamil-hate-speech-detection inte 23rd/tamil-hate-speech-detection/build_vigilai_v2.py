"""
Builds vigilai_v4_full_2.csv — a corrected copy of vigilai_v4_full.csv.

SCOPE (per explicit instruction): dataset file only. Does NOT import or
modify tamil_slur_lexicon.py, hate_detector.py, or anything under api/.
This script keeps its own local, throwaway word lists purely to decide
which hate-labelled rows are mislabeled — none of that touches the
production lexicon.

Only direction checked: label=1 (hate) rows that should actually be
label=0 (safe). Safe-labelled rows are left untouched (not in scope).

Lesson baked in from a previous bad attempt: a naive "contains a positive
word" check is unsafe — it gets fooled by sarcasm and by negated
compounds like நன்றி கெட்ட ("ungrateful", which literally contains
நன்றி/"thanks" as a substring but means the opposite). So a flip only
happens when ALL of these hold:

  1. reason == 'source_labelled_hate'
  2. no Tier-1 profanity substring match (local copy of the project's
     own T1 list, plus a few additional slur spellings it was missing,
     e.g. தேவஅடியாள் - a variant spelling not in tamil_slur_lexicon.py)
  3. no match against an expanded "negative signal" list covering
     indirect/contextual hate markers (தவறு, இயற்கைக்கு மாறான, வெறி,
     ஒழி, அழி, கொலை, தூக்கு, செருப்பால், நாய், பன்றி, கெட்ட, இழிவு,
     அசிங்கம், மிருகம், ஆபாச, etc.) found ANYWHERE in the text — this
     is what catches "நன்றி கெட்ட" (blocks the flip, correctly) and
     "...நாய்களா" (blocks the flip, correctly)
  4. contains a clear, unambiguous praise/neutral marker
  5. word_count <= 10 — short plain compliments are the real pattern
     found during manual review (e.g. "Very good speech god bless you
     my sister", "Good human being"); longer multi-clause rows are
     where sarcastic/contextual hate hides and are NOT auto-flipped
     even if they contain an isolated positive-looking word near the end

Everything that doesn't clear all 5 stays at its original label.
Nothing is deleted — vigilai_v4_full.csv is untouched; this writes a
new file alongside it.
"""
import pandas as pd

SRC = "data/text/vigilai_v4_full.csv"
OUT = "data/text/vigilai_v4_full_2.csv"
AUDIT = "D:/IntegrateV2/vigilai_v2_build_audit.txt"

# Local-only slur check (NOT the production lexicon). Deliberately a
# superset of tamil_slur_lexicon.py's T1 list plus spellings it missed.
T1_SLURS = [
    "தேவடிய", "தேவிடிய", "தேவடியா", "தேவுடிய", "தேவஅடியாள்", "தேவடியாள்",
    "புண்ட", "புண்டை", "புன்ட", "ஓத்தா", "தாயோளி", "தாயோலி", "கூதி",
    "சுன்னி", "சுண்ணி", "ஊம்பு", "ஊம்ப", "ஊம்பி", "பூலு", "எச்சப்பய",
    "எச்சப்புன்ட", "ஈனப்பிறவி", "கேனப்பய", "கேனப்பிறவி", "கூமுட்ட",
    "கொய்யால", "பொட்ட", "வேசி",
    "thevdiya", "thevidiya", "thevadiya", "punda", "pundai", "otha",
    "ootha", "thayoli", "koothi", "sunni", "oombu", "oomba", "poolu",
    "fuck", "fucking", "fucker", "bitch", "bastard", "asshole", "cunt",
]

# Broader "this sentence is doing something negative/contemptuous"
# signal — ANY presence anywhere blocks a flip. Intentionally wide,
# because a missed flip just leaves the original label (safe failure
# mode); a wrong flip is the thing to avoid.
NEGATIVE_SIGNALS = [
    "தவறு", "தப்பு", "அருவருப்பு", "அருவருக்க", "கேவலம", "வெறி", "ஒழிக்க",
    "ஒழிய", "ஒழிந்து", "அழிய", "அழிக்க", "அழிவு", "கொலை", "தூக்கில்",
    "தூக்கு", "செருப்பால", "செருப்படி", "நாய்", "பன்றி", "கழுதை", "கெட்ட",
    "இழிவு", "இழிந்த", "இழிபிறவி", "அசிங்க", "மிருகம", "விலங்கு", "ஆபாச",
    "முட்டாள", "பைத்தியம", "நாறி", "நாற்றம", "காட்டுமிராண்ட", "அவமான",
    "சனியன", "ஈனம", "குண்ட", "லூசு", "மானங்கெட்ட", "கும்பல", "இயற்கைக்கு மாறான",
    "இயற்கைக்கு எதிரான", "கொண்டான", "சாபம", "வேண்டாம", "தீ", "சீ ", "தூ ",
    "stupid", "idiot", "moron", "worthless", "clown", "rascal", "shame",
    "disgust", "bastard",
]

# IMPORTANT, found by manual audit of a first attempt: Tamil "positive"
# words (நல்ல/வாழ்த்து/பாராட்டு etc.) are NOT a safe signal in this
# dataset — they're used sarcastically far more often than literally
# (e.g. "நல்ல வளர்ப்பு" after an insult, "நல்ல பாடம்" = wishing someone
# a punishment, "நல்லா கூட்டிக் கொடுப்பாரு" = an idiom for pimping).
# Every Tamil-triggered flip in that first attempt was wrong; every
# English-triggered one was correct (plain English compliments that
# clearly inherited "hate" from their thread, not their own content).
# So this list is deliberately English-only — no Tamil words — for the
# auto-flip decision.
POSITIVE_MARKERS = [
    "very good", "god bless", "bold like her", "matured speech",
    "good human being", "supr speech", "super speech",
    "well done", "congratulat", "thank you", "thanks", "beautiful",
    "wonderful", "nice video", "great job", "best wishes",
    "good morning", "proud of you", "👌👌👌",
]


def has_any(text, words):
    # English phrases in POSITIVE_MARKERS are case-insensitive by nature
    # (e.g. "Good human being" must match "good human being"); Tamil
    # script has no case so .lower() is a no-op for it.
    low = text.lower()
    return any(w.lower() in low for w in words)


def main():
    df = pd.read_csv(SRC)
    df["text"] = df["text"].astype(str)

    flipped_rows = []
    final_label = df["label"].copy()

    for i, row in df.iterrows():
        if row["reason"] != "source_labelled_hate":
            continue
        text = row["text"]
        wc = row["word_count"]

        if has_any(text, T1_SLURS):
            continue
        if has_any(text, NEGATIVE_SIGNALS):
            continue
        if not has_any(text, POSITIVE_MARKERS):
            continue
        if wc > 10:
            continue

        final_label[i] = 0
        flipped_rows.append((i, text, wc))

    out_df = df.copy()
    out_df["label"] = final_label
    out_df.to_csv(OUT, index=False, encoding="utf-8-sig")

    with open(AUDIT, "w", encoding="utf-8") as f:
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"source_labelled_hate rows checked: {(df['reason']=='source_labelled_hate').sum()}\n")
        f.write(f"Rows flipped hate->safe: {len(flipped_rows)}\n\n")
        f.write("Original label counts:\n")
        f.write(str(df["label"].value_counts()) + "\n\n")
        f.write("New label counts (vigilai_v4_full_2):\n")
        f.write(str(out_df["label"].value_counts()) + "\n\n")
        f.write("=== ALL flipped rows (full text, for manual spot-check) ===\n\n")
        for idx, text, wc in flipped_rows:
            f.write(f"[row {idx}, wc={wc}] {text}\n\n")

    print(f"done, flipped={len(flipped_rows)}")


if __name__ == "__main__":
    main()

# =============================================================
# TEXT PREPROCESSING MODULE
# Extracted from: notebooks/final text module.ipynb
# 
# DO NOT MODIFY — must match training preprocessing exactly.
# =============================================================

import re


# =============================================================
# HATE SPEECH WORD DICTIONARY
# The ONLY words that get converted to Unicode.
# Everything else — Sinhala romanized, English, mixed — stays
# exactly as typed.
# =============================================================

HATE_WORDS = {

    # ── Direct insults and slurs ──────────────────────────────
    "huththo"       : "හුත්තෝ",
    "hutto"         : "හුත්තෝ",
    "hutho"         : "හුත්තෝ",
    "huththaa"      : "හුත්තා",
    "huththa"       : "හුත්ත",
    "pakaya"        : "පාකයා",
    "paka"          : "පාක",
    "pakayaa"       : "පාකයා",
    "modaya"        : "මෝඩයා",
    "moda"          : "මෝඩ",
    "modai"         : "මෝඩයි",
    "modayaa"       : "මෝඩයා",
    "pissu"         : "පිස්සු",
    "pissa"         : "පිස්සා",
    "pissuda"       : "පිස්සුද",
    "pissuwek"      : "පිස්සුවෙක්",
    "yakka"         : "යක්කා",
    "yako"          : "යකෝ",
    "balla"         : "බල්ලා",
    "ballaa"        : "බල්ලා",
    "ballanta"      : "බල්ලන්ට",
    "godaya"        : "ගොදයා",
    "godayaa"       : "ගොදයා",
    "kunu"          : "කුණු",
    "kunuharupa"    : "කුණුහරුප",
    "kunuharuwa"    : "කුණුහරුව",
    "baiya"         : "බෙයියා",
    "bayya"         : "බය්යා",
    "hora"          : "හොර",
    "horakam"       : "හොරකම්",
    "horayaa"       : "හොරයා",
    "kela"          : "කේළ",
    "kattiya"       : "කට්ටිය",

    # ── Targeting words (used when directing hate at someone) ─
    "umbata"        : "ඔඹට",
    "umba"          : "ඔඹ",
    "umbala"        : "ඔඹලා",
    "umbatama"      : "ඔඹටම",

    # ── Derogatory group labels ───────────────────────────────
    "demalu"        : "දෙමළු",
    "demalunta"     : "දෙමළුන්ට",
    "muslimayya"    : "මුස්ලිම්අය්යා",
    "kollantar"     : "කොල්ලන්තාර්",

    # ── Sexual / explicit hate terms ─────────────────────────
    "keli"          : "කෙළි",
    "wesige"        : "වේශිගේ",
    "vesi"          : "වේශී",
    "wesiya"        : "වේශියා",
    "hukana"        : "හූකන",
    "hukanawa"      : "හූකනවා",

    # ── Threatening / violent language ───────────────────────
    "maranawa"      : "මරනවා",
    "marapan"       : "මරාපන්",
    "gahanna"       : "ගහන්න",
    "kapanna"       : "කාපන්න",
    "nasanawa"      : "නසනවා",

    # ── Common hate speech qualifiers ────────────────────────
    "naraka"        : "නරක",
    "narakai"       : "නරකයි",
    "narak"         : "නරක",
    "apahu"         : "අපහු",
    "nidahas"       : "නිදහස්",
}

# Spelling variations of hate words → canonical form
HATE_VARIATIONS = {
    # huththo variants
    "huthoo"    : "huththo",
    "hutho"     : "huththo",
    "huttho"    : "huththo",
    "huththa"   : "huththo",
    "huththoo"  : "huththo",

    # pakaya variants
    "pakayaa"   : "pakaya",
    "pakayyaa"  : "pakaya",

    # modaya variants
    "modayaa"   : "modaya",
    "moodaya"   : "modaya",
    "moodayaa"  : "modaya",

    # pissu variants
    "pissoo"    : "pissa",
    "pissaa"    : "pissa",

    # balla variants
    "ballaa"    : "balla",
    "balloo"    : "balla",

    # umba variants
    "umbata"    : "umbata",
    "umbataa"   : "umbata",

    # kunu variants
    "kunuu"     : "kunu",
    "kuunu"     : "kunu",

    # hora variants
    "horaa"     : "hora",
    "hoorana"   : "hora",
}


# =============================================================
# GENERAL TEXT CLEANER
# =============================================================

def clean_text(text):
    """
    General-purpose cleaner for social media Sinhala text.
    Safe to run on Unicode Sinhala — does not touch Sinhala chars.
    Safe to run on Romanized Sinhala — removes noise only.
    """
    text = str(text).strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # real URLs
    text = re.sub(r'\bURL\b', '', text)                  # placeholder token
    text = re.sub(r'@\w+', '', text)                     # remove mentions
    text = re.sub(r'#\w+', '', text)                     # hashtags
    text = re.sub(r'\b\d+\b', '', text)                  # standalone numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================
# NOISE CLEANER
# =============================================================

def clean_noise(text):
    """Fix repeated characters common in social media typing."""
    text = text.lower().strip()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)   # keep max 2 repeats
    text = re.sub(r'\s+', ' ', text)
    return text


# =============================================================
# SCRIPT DETECTOR
# =============================================================

def is_romanized(text):
    """
    Returns True if less than 10% of non-space characters are
    Unicode Sinhala — meaning the text is Romanized Sinhala.
    """
    text = str(text)
    sinhala_chars = sum(1 for c in text if '\u0D80' <= c <= '\u0DFF')
    total = len(text.replace(' ', ''))
    if total == 0:
        return False
    return (sinhala_chars / total) < 0.1


# =============================================================
# CORE: TARGETED HATE WORD CONVERTER
# =============================================================

def convert_hate_words(text):
    """
    Scan text word by word.
    Hate speech words → Unicode Sinhala.
    Everything else (Sinhala romanized, English, mixed) → unchanged.
    """
    words  = text.split()
    result = []

    for word in words:
        # Detach trailing punctuation so "hutto!" still matches "hutto"
        punct = ""
        if word and word[-1] in ".,!?;:\"'":
            punct = word[-1]
            word  = word[:-1]

        if not word:
            result.append(punct)
            continue

        w = word.lower().strip()

        # Step 1: check spelling variation → normalise to canonical
        canonical = HATE_VARIATIONS.get(w, w)

        # Step 2: check canonical form against hate word dictionary
        if canonical in HATE_WORDS:
            result.append(HATE_WORDS[canonical] + punct)
        else:
            # Not a hate word — keep exactly as typed
            result.append(word + punct)

    return " ".join(result)


# =============================================================
# FULL PREPROCESSING PIPELINE
# Entry point for inference.
# =============================================================

def preprocess(text):
    """
    Full preprocessing pipeline.

    Input  : Any Sinhala social media text (Unicode or Romanized)
    Output : (processed_text, script_type)

    Steps:
      1. General cleaning — URLs, @USER, URL token, numbers, hashtags
      2. Script detection — is it Romanized?
      3. If Romanized:
           a. Noise cleaning — fix hondaaaa → honda
           b. Hate word conversion — huththo → හුත්තෝ, umbata → ඔඹට
              (everything else stays as typed)
      4. If Unicode — return as-is after cleaning
    """
    text = str(text).strip()

    # Step 1: general cleaning (both script types)
    cleaned = clean_text(text)

    if is_romanized(cleaned):
        # Step 2a: fix repeated chars
        cleaned = clean_noise(cleaned)
        # Step 2b: convert only hate words to Unicode
        final = convert_hate_words(cleaned)
        return final, "romanized"

    return cleaned, "unicode"

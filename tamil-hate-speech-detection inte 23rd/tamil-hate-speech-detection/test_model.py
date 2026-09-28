"""
Run this FIRST in VS Code.
    python test_model.py

Checks the model on your known failure cases before you touch the pipeline.
"""

from hate_detector import HateDetector

MODEL_PATH = "models/text_models/vigilai_xlmr_clean"   # <-- change if your folder is different

d = HateDetector(MODEL_PATH)

SHOULD_BE_SAFE = [
    "காரமா வேணுமா",
    "ஒரு மரம் ஆயிரம் தீக்குச்சிகளை உருவாக்கும் ஆனால் ஒரு தீக்குச்சி ஆயிரம் மரங்களை எரித்துவிடும்",
    "அதனால ஃபர்ஸ்ட் மார்க் வாங்குற பிள்ளைய பிடிக்கும்னு நினைக்காத",
    "Iron man fans dis like podunga",
    "மீன் குழம்பு வைக்கறதுக்கு முதல்ல வெங்காயம் நல்லா வதக்கணும். அப்புறம் புளி கரைச்சு ஊத்தணும்.",
    "ராவணன் ஒரு கெட்டவன் இல்லை. அவன் ஒரு பெரிய அறிஞன். சீதைய கடத்துனது தப்பு தான்.",
    "அந்த ஐநூறு ரூபாய் நோட்டு போலியானது. கடைசில எல்லாரும் நஷ்டப்பட்டாங்க.",
    "என் புருஷன் என்ன அடிச்சாரு. ஆனாலும் நான் அவரோட தான் இருக்கேன்.",
    "அங்கன்வாடி ஊழியர் ஒருத்தர் குழந்தைய துன்புறுத்துன வழக்குல கைது பண்ணப்பட்டிருக்காரு.",
    "மரம் வெட்டுறத தடுக்கணும். சுற்றுச்சூழலுக்கு நல்லது இல்ல.",
    "நாய் குரைக்குது. வெளியே யாரோ வந்திருக்காங்க போல.",
    "சாதி பாகுபாடு ஒழிய வேண்டும் என்று அம்பேத்கர் வாழ்நாள் முழுக்க போராடினார்.",
    "வெங்காயம் பொடியா வெட்டி வையுங்க. அப்புறம் எண்ணெய்ல வதக்கணும்.",
    "கொலை வழக்குல ஒருத்தர கைது பண்ணிருக்காங்க. விசாரணை நடக்குது.",
    "இட்லி சாம்பார் நல்லா இருந்துச்சு",
]

SHOULD_BE_HATE = [
    "போடா தேவிடியா மகனே",
    "நீ ஒரு மடையன். உன் முகத்த பார்க்கவே கேவலமா இருக்கு.",
    "உன்ன வெட்டி கொல்லுவேன்டா",
    "You are a bastard, get lost",
    "இந்த ஜாதியினர் எல்லாம் கேவலமானவங்க. நம்ம கூட வர்றதுக்கே தகுதியில்ல.",
    "இந்த மதத்தினர் நம்பகமில்லாதவங்க. நம்மோட மதம் மட்டுமே சரியான மதம்.",
]

print("\n" + "=" * 80)
print("SHOULD BE SAFE")
print("=" * 80)
fa = 0
for t in SHOULD_BE_SAFE:
    r = d.check(t)
    bad = r["verdict"] == "HATE"
    fa += bad
    mark = "  <-- FALSE ALARM" if bad else ""
    print(f"{r['verdict']:10} {r['confidence']*100:5.1f}%  [{r['source']:8}]  {t[:48]}{mark}")

print("\n" + "=" * 80)
print("SHOULD BE HATE")
print("=" * 80)
ms = 0
for t in SHOULD_BE_HATE:
    r = d.check(t)
    bad = r["verdict"] != "HATE"
    ms += bad
    mark = "  <-- MISSED" if bad else ""
    print(f"{r['verdict']:10} {r['confidence']*100:5.1f}%  [{r['source']:8}]  {t[:48]}{mark}")

print("\n" + "=" * 80)
print(f"FALSE ALARMS: {fa} / {len(SHOULD_BE_SAFE)}      (want 0-1)")
print(f"MISSED HATE : {ms} / {len(SHOULD_BE_HATE)}      (want 0-2)")
print("=" * 80)

# ---------- chunk test ----------
print("\nCHUNK TEST — a full 30s cooking transcript")
cooking = ("இன்னைக்கு மீன் குழம்பு பண்ணப்போறோம். முதல்ல வெங்காயம் நல்லா பொடியா வெட்டணும். "
           "அப்புறம் எண்ணெய்ல வதக்கணும். தக்காளி சேர்த்து நல்லா வேகவிடணும். "
           "புளி கரைச்சு ஊத்திட்டு கொதிக்க வைக்கணும். மீன் போட்டு பத்து நிமிஷம் வேகவிடுங்க. "
           "காரம் வேணும்னா மிளகாய் தூள் கூட்டி போடுங்க.")
r = d.check_chunk(cooking)
print(f"  verdict   : {r['verdict']}   (want SAFE)")
print(f"  sentences : {r['n_sentences']}   flagged: {r['n_flagged']}")
print(f"  worst     : {r['worst_confidence']*100:.1f}%  ->  {r['worst_sentence'][:55]}")

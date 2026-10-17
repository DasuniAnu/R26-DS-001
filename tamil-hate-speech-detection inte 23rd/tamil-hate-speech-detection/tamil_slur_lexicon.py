import re

# TIER 1 — unambiguous profanity. Presence alone = HATE.
T1_TA = ['தேவடிய','தேவிடிய','தேவடியா','புண்ட','புண்டை','புன்ட','ஓத்தா','தாயோளி','தாயோலி',
         'கூதி','சுன்னி','சுண்ணி','ஊம்பு','ஊம்ப','ஊம்பி','பூலு','எச்சப்பய','எச்சப்புன்ட',
         'ஈனப்பிறவி','கேனப்பய','கேனப்பிறவி','கூமுட்ட','கொய்யால','பொட்ட','வேசி','தேவுடிய']
T1_EN = ['thevdiya','thevidiya','thevadiya','punda','pundai','pundiyandi','otha','ootha',
         'thayoli','thaioli','koothi','sunni','oombu','oomba','poolu','koomuta','myru','mayiru',
         'fuck','fucking','fucker','bitch','bastard','bastad','asshole','motherfucker','cunt','dickhead']

# TIER 2 — insults. Only HATE when aimed at someone.
T2_TA = ['நாயே','பன்னி','கழுதை','மடையன்','மடைய','மோடயன்','லூசு','கோமாளி','முட்டாள்','குண்டே',
         'நொக்காள','பரதேசி','அயோக்கிய','தரங்கெட்ட','கேவலமான','நீசன்','காட்டுமிராண்டி']
T2_EN = ['madayan','madaiya','loosu','naaye','panni','kazhudai','kunde','nokkaala','stupid',
         'idiot','moron','worthless','rascal']

# ATTACK CONTEXT — second person / vocative
CTX_TA = ['நீ ','நீங்க','உன்','உங்கள','உனக்','உன்ன','எலே','ஏண்டா','ஏண்டி','எவன்','இவன்',
          'போடா','போடி','டேய்','மவனே','மகனே','மவளே','நாயே','உங்கம்மா','உன்னோட']
CTX_VOC = re.compile(r'$^')  # disabled: Tamil verb endings collide with vocatives
CTX_EN = ['nee','unna','undan','poda','podi','dei','dai','yenda','ennada','avan','ivan','u r','you are']

# THREATS
THR_TA = ['கொல்லு','கொன்ன','கொல்வ','கொல்ல','வெட்டு','வெட்டி','செருப்பால','செருப்படி',
          'எரிச்சு','ஒழிச்சு','அழிச்சு','விரட்டு','சாகடி','செத்துடு','தொலை']
THR_EN = ['kollunga','kollu','vettu','saagadi','get lost']

def _ta(words):   # Tamil: plain substring is safe for these stems
    return re.compile('|'.join(map(re.escape, words)))
def _en(words):   # Latin: MUST use word boundaries
    return re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')', re.IGNORECASE)

R1_TA, R1_EN = _ta(T1_TA), _en(T1_EN)
R2_TA, R2_EN = _ta(T2_TA), _en(T2_EN)
RC_TA, RC_EN = _ta(CTX_TA), _en(CTX_EN)
RT_TA, RT_EN = _ta(THR_TA), _en(THR_EN)

def has_t1(t):  return bool(R1_TA.search(t) or R1_EN.search(t))
def has_t2(t):  return bool(R2_TA.search(t) or R2_EN.search(t))
def has_ctx(t): return bool(RC_TA.search(t) or RC_EN.search(t) or CTX_VOC.search(t))
def has_thr(t): return bool(RT_TA.search(t) or RT_EN.search(t))

def classify(t):
    t = str(t)
    if has_t1(t):                 return 1, 't1_profanity'
    if has_thr(t) and has_ctx(t): return 1, 'threat_directed'
    if has_t2(t) and has_ctx(t):  return 1, 'insult_directed'
    return 0, 'clean'

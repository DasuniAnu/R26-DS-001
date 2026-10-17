"""
VigilAI — Tamil hate speech detector
Two layers: word list first, then the model.

Usage:
    from hate_detector import HateDetector
    d = HateDetector("models/vigilai_xlmr_clean")
    print(d.check("காரமா வேணுமா"))
"""

import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tamil_slur_lexicon import classify as lexicon_check


class HateDetector:
    def __init__(self, model_path, hate_threshold=0.75, uncertain_threshold=0.45):
        print(f"loading model from {model_path} ...")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.hate_th = hate_threshold
        self.unc_th = uncertain_threshold
        print(f"ready on {self.device}")

    # ---------- model score for one sentence ----------
    def _model_prob(self, text):
        enc = self.tok(text, truncation=True, padding=True,
                       max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return F.softmax(logits, dim=1)[0][1].item()

    # ---------- ONE sentence ----------
    def check(self, text, use_lexicon=True):
        text = str(text).strip()
        if len(text) < 2:
            return {"verdict": "SAFE", "confidence": 0.0, "source": "too_short"}

        # LAYER 1 — word list. If a real slur is present, done. No model needed.
        # use_lexicon=False skips this layer entirely, for testing the model alone.
        if use_lexicon:
            lex, reason = lexicon_check(text)
            if lex == 1:
                return {"verdict": "HATE", "confidence": 0.973,
                        "source": "lexicon", "reason": reason}

        # LAYER 2 — model
        p = self._model_prob(text)
        if p >= self.hate_th:
            verdict = "HATE"
        elif p >= self.unc_th:
            verdict = "UNCERTAIN"
        else:
            verdict = "SAFE"

        return {"verdict": verdict, "confidence": round(p, 4),
                "source": "model", "reason": "model_score"}

    # ---------- split a transcript into sentences ----------
    @staticmethod
    def split_sentences(text, min_words=2):
        parts = re.split(r"[.?!\n।]+", str(text))
        return [p.strip() for p in parts if len(p.strip().split()) >= min_words]

    # ---------- ONE 30-second chunk ----------
    def check_chunk(self, transcript, strong_threshold=0.93, use_lexicon=True):
        """
        A chunk is HATE only if:
          - a real slur is found, OR
          - 1 sentence is flagged very strongly (>= 0.93)

        Everything else is SAFE (no "2+ weaker sentences" rule, no UNCERTAIN
        middle tier for video segments) — YouTube pipeline only.
        """
        sentences = self.split_sentences(transcript)
        if not sentences:
            return {"verdict": "SAFE", "flagged": [], "n_sentences": 0}

        results = [(s, self.check(s, use_lexicon=use_lexicon)) for s in sentences]
        flagged = [(s, r) for s, r in results if r["verdict"] == "HATE"]

        has_slur = any(r["source"] == "lexicon" for _, r in flagged)
        has_strong = any(r["confidence"] >= strong_threshold for _, r in flagged)

        verdict = "HATE" if (has_slur or has_strong) else "SAFE"

        worst = max(results, key=lambda x: x[1]["confidence"])
        return {
            "verdict": verdict,
            "n_sentences": len(sentences),
            "n_flagged": len(flagged),
            "flagged": [{"text": s, **r} for s, r in flagged],
            "worst_sentence": worst[0],
            "worst_confidence": worst[1]["confidence"],
        }

    # ---------- whole video ----------
    def check_video(self, chunks, hate_chunk_ratio=0.15):
        """chunks = list of transcript strings, one per 30s segment"""
        out = [self.check_chunk(c) for c in chunks]
        n_hate = sum(1 for r in out if r["verdict"] == "HATE")
        ratio = n_hate / max(1, len(out))
        return {
            "verdict": "HATE SPEECH DETECTED" if ratio >= hate_chunk_ratio else "VIDEO IS SAFE",
            "hate_chunks": n_hate,
            "total_chunks": len(out),
            "ratio": round(ratio, 3),
            "chunks": out,
        }

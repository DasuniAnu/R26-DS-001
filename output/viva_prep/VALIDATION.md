# Guide verification — 14 October 2026

## Source coverage

- Read all eight application/test Python files across backend and frontend, both requirements files, ASR notes, Streamlit configuration and relevant local checkpoint configurations.
- Generated an AST inventory of 58 function/method definitions, including nested report helpers and test helpers.
- Verified that every function name appears in the explanatory guide.
- Included complete numbered source for 17 files in SOURCE_ATLAS.html.
- Parsed all eight Python files successfully and verified source/configuration hashes after document generation. No application edits were made for this task.
- Inspected selected training-related cells/script sections to establish provenance limits; did not certify all historical training runs or their outputs.

## Regression verification

Command run from project root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s backend -p test_asr.py
```

Result: **Ran 10 tests in 11.241s — OK**.

The logged RuntimeError “test failure” was deliberately injected by the ASR failure-handling test and did not represent a failed test. These tests mock heavy model/network dependencies; they do not validate real-world model accuracy.

## Document checks

- Main guide: 22 chapters and 44 numbered mock viva questions.
- Generated HTML has unique anchor IDs and all internal anchor targets resolve.
- Function atlas links resolve to the corresponding source lines.
- Files are UTF-8 and use local font fallbacks; no external JavaScript or stylesheet is required.
- Browser discovery found no available browser. A rendered visual review and actual Print-to-PDF check could not be completed. HTML was checked structurally, not claimed to be visually verified.

## Evidence not established by this review

No new full-video inference benchmark, WER/CER evaluation, threshold calibration, classifier accuracy measurement or training provenance certification was performed. The earlier conversation documented a successful real-video smoke test on 2 October; that is separate from the fresh offline regression run above.

Call references in the source atlas are static syntactic matches, not dynamic traces. Framework callbacks and ambiguous repeated method names are labelled accordingly. Line numbers and hashes describe the reviewed snapshot and should be refreshed if code changes.

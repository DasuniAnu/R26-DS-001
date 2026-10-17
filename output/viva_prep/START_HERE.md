# Your viva study pack

Open **VIVA_GUIDE.html** in a browser. It has a chapter sidebar, code examples, 44 mock questions, a glossary, demonstration steps and a Print / Save as PDF button. Keep SOURCE_ATLAS.html in the same folder so the companion link works.

## Files

- VIVA_GUIDE.html: the main readable guide, 22 chapters and approximately 14,500 words.
- VIVA_GUIDE.md: editable text version, readable directly in VS Code.
- SOURCE_ATLAS.html: all 17 reviewed application/configuration files, line-numbered source, 58 function/method definitions and syntactic local call references.
- VALIDATION.md: test result and review limits.
- source_manifest.json: SHA-256 hashes of the source snapshot.
- function_inventory.json: machine-readable function locations.
- build_guide.py: optional stdlib-only HTML/atlas builder; running it refreshes the atlas from the current workspace, but does not automatically rewrite the explanatory prose.

## Study order

1. Chapters 2–5: explain the system without looking at code.
2. Chapters 6–11: trace preprocessing, models, YouTube, ASR and routes.
3. Chapters 12–15: explain frontend behavior and acknowledge the verified limitations.
4. Chapter 16: practise explaining each code pattern aloud.
5. Chapters 17–18: distinguish training evidence from inference and testing.
6. Chapter 19: answer the 44 questions, then check the answers.
7. Chapter 20: rehearse the live demonstration.

For every function, practise: purpose → input → operations → output → caller → limitation. Use Ctrl+F in the atlas to locate its definition, then click a line number to bookmark that location.

The current deepfake checkpoint is WavLM. Some old UI labels say Wav2Vec2. ASR confidence is not transcription accuracy. The exported report currently loses some uncertainty labels. These are important viva discussion points explained in detail rather than hidden.

The application was not changed to prepare these documents. The guide contains actual behavior, known limitations and proposed evaluation work, clearly separated. It does not invent final training metrics or claim measured ASR accuracy gains.

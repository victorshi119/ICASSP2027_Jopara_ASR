# Table 1: CER / WER (%) on the 150-utterance test set, corpus-level sclite, no post-processing

Rows 8 and 9: mean ± sample standard deviation over router seeds 13, 42, 123.

| Row | System | Guaraní | Inter-sentential | Jehe'a | Spanish | Overall |
|---|---|---|---|---|---|---|
| 1 | Whisper-large-v3, automatic language choice | 89.0 / 128.2 | 45.6 / 62.4 | 55.9 / 79.7 | 20.3 / 42.3 | 47.4 / 66.7 |
| 2 | MMS-1B-all, Guaraní adapter (no auto-detect) | 30.0 / 82.2 | 34.1 / 78.6 | 30.8 / 76.8 | 26.4 / 64.8 | 31.6 / 75.7 |
| 3 | OmniASR, auto-detect mode (baseline) | 41.6 / 93.3 | 30.4 / 50.9 | 30.0 / 58.2 | 14.4 / 25.9 | 28.4 / 50.4 |
| 4a | OmniASR, Guaraní language mode (grn_Latn) | 35.1 / 82.8 | 26.9 / 48.1 | 26.0 / 56.6 | 13.5 / 25.8 | 25.0 / 47.9 |
| 4b | OmniASR, Paraguayan Guaraní language mode (gug_Latn) | 37.4 / 89.0 | 28.8 / 52.5 | 28.4 / 61.0 | 14.8 / 30.3 | 27.0 / 52.4 |
| 4c | OmniASR, Spanish language mode (spa_Latn) | 48.0 / 105.5 | 30.7 / 51.8 | 31.6 / 61.5 | 14.4 / 25.8 | 29.5 / 52.3 |
| 5 | Spanish adapter | 55.7 / 119.6 | 39.6 / 60.6 | 45.3 / 84.9 | 12.4 / 24.8 | 37.7 / 63.2 |
| 6 | Guaraní adapter | 32.2 / 87.7 | 29.1 / 51.2 | 37.1 / 70.7 | 15.0 / 28.6 | 29.1 / 53.8 |
| 7 | Code-switching adapter | 32.7 / 80.4 | 32.7 / 52.3 | 34.9 / 68.8 | 11.8 / 25.9 | 29.7 / 52.9 |
| 8 | Mixture of experts over all three adapters | 31.8±1.0 / 83.4±0.6 | 32.4±2.1 / 51.0±2.6 | 32.4±4.1 / 64.7±5.8 | 11.5±0.1 / 22.9±0.3 | 28.8±2.0 / 50.9±2.8 |
| 9 | Mixture of experts, code-switching adapter left empty | 32.6±0.3 / 83.0±2.0 | 25.7±0.1 / 45.3±0.3 | 27.6±0.2 / 57.4±0.1 | 11.4±0.2 / 22.2±0.3 | 24.3±0.1 / 46.1±0.2 |

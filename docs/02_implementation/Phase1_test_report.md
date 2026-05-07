# Phase 1 Full Test Report

Generated from:

```text
outputs/phase1_full_test/latest/phase1_full_test_metrics.json
outputs/phase1_full_test/latest/buffer_metrics.csv
outputs/phase1_full_test/latest/frame_metrics.csv
```

Run window:

```text
started_utc: 2026-05-06T22:56:36.062537Z
completed_utc: 2026-05-07T01:22:17.016887Z
```

## Executive Summary

The full six-night Phase 1 desktop mock completed successfully. The run
processed all expected audio buffers, wrote the expected edge rows and vectors,
synced every pending detection to the hub, and finished with a healthy watchdog
status.

The software pipeline is behaving correctly as a mock system test. Every
15-second buffer produced one edge event and three 5-second embedding rows.
Every edge row was accepted by the hub and marked `synced` only after hub
acknowledgement.

The main tuning finding is that the current first-pass biological gate is far
too permissive for storage reduction. All 16,560 buffers were retained as
`bio_hit`, mostly because broad Perch labels such as `Animal` and
`Wild_animals` exceeded the configured biological threshold. This proves the
end-to-end retention path, but it does not yet provide a useful field retention
policy.

## Test Input

| Item | Value |
| --- | --- |
| Raw audio root | `/data/petrel_acoustics/raw_audio/doc_ar4/rapanui_AR4_june_2023` |
| Night folders processed | 6 |
| Audio files processed | 277 |
| Audio duration | 69.0 hours |
| Buffer size | 15 seconds |
| Expected buffers | 16,560 |
| Processed buffers | 16,560 |
| Perch frames | 49,680 |
| Include partial final buffer | `false` |
| Device ID | `pi_01` |
| Perch model | `perch_v2_cpu/1` from TensorFlow Hub/Kaggle |
| Model load time | 3.08 seconds |

## Pipeline Integrity

| Check | Result |
| --- | ---: |
| Edge `buffer_events` rows | 16,560 |
| Edge `embedding_segments` rows | 49,680 |
| Edge vector rows | 49,680 |
| Edge pending rows after sync | 0 |
| Edge synced rows after sync | 16,560 |
| Hub ingestion batches | 166 |
| Hub `hub_buffer_events` rows | 16,560 |
| Hub `hub_embedding_segments` rows | 49,680 |
| Hub vector rows | 49,680 |
| Hub health metric rows | 166 |
| Watchdog status | `healthy` |

The expected invariant held throughout the run:

```text
1 buffer event = 3 Perch frames = 3 embedding rows = 3 vector rows
```

The sender also behaved correctly. It sent 166 batches, the hub accepted
16,560 buffer IDs, and the edge database had no pending rows after the sync.

## Speed And Performance

| Metric | Value |
| --- | ---: |
| Total wall time | 2.43 hours |
| Buffer processing time | 2.40 hours |
| Hub sync time | 54.95 seconds |
| Total inference time | 8,246.49 seconds |
| Mean inference time per 15-second buffer | 0.498 seconds |
| Median inference time per 15-second buffer | 0.495 seconds |
| P95 inference time per 15-second buffer | 0.522 seconds |
| Inference real-time factor | 30.12x |
| Overall wall-clock factor | 28.42x |
| Buffers processed per second | 1.92 |
| DB write time | 188.13 seconds |
| FLAC save time | 191.45 seconds |

The desktop mock is comfortably faster than real time on this machine. The
Perch inference path processed 69 hours of audio in about 2.29 hours of
inference time, roughly 30x real time. Including database writes, FLAC writes,
and hub sync, the full wall-clock run remained about 28x real time.

Storage and transport output:

| Artifact | Size / Count |
| --- | ---: |
| Retained FLAC files | 16,560 |
| Retained FLAC size | 9.02 GiB |
| Edge SQLite DB | 330 MiB |
| Hub SQLite DB | 330 MiB |
| MessagePack payload total | 323.29 MiB |
| MessagePack per buffer | 19.99 KiB |
| Buffer metrics CSV | 6.9 MiB |
| Frame metrics CSV | 33 MiB |

The MessagePack payload size is consistent with three 1,536-dimensional
float32 embeddings per 15-second buffer, plus metadata and telemetry.

## Nightly Summary

| Night folder | Audio hours | Files | Buffers | Bio-hit buffers | Noise-dominant buffers | Noise-dominant buffer rate | Inference real-time factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20230527` | 12.0 | 48 | 2,880 | 2,880 | 2,015 | 70.0% | 30.17x |
| `20230528` | 12.0 | 48 | 2,880 | 2,880 | 2,396 | 83.2% | 29.67x |
| `20230529` | 12.0 | 48 | 2,880 | 2,880 | 2,370 | 82.3% | 29.85x |
| `20230530` | 12.0 | 48 | 2,880 | 2,880 | 2,459 | 85.4% | 30.33x |
| `20230531` | 12.0 | 48 | 2,880 | 2,880 | 2,599 | 90.2% | 30.46x |
| `20230601` | 9.0 | 37 | 2,160 | 2,160 | 2,024 | 93.7% | 30.31x |

Noise dominance increased across the later nights, especially the final
9-hour folder. This matches the configured noise labels frequently activating
alongside the broad biological labels.

## Retention And Gate Activations

| Gate / outcome | Count | Rate |
| --- | ---: | ---: |
| Retained as `bio_hit` | 16,560 buffers | 100.0% |
| Retained as `validation_sample` | 0 buffers | 0.0% |
| Dropped | 0 buffers | 0.0% |
| Biological gate active | 16,560 buffers | 100.0% |
| Biological gate active | 49,679 frames | 100.0% |
| Noise dominant | 13,863 buffers | 83.7% |
| Noise dominant | 28,267 frames | 56.9% |

Top biological labels at the buffer level:

| Label | Buffers |
| --- | ---: |
| `Animal` | 10,879 |
| `Wild_animals` | 5,674 |
| `Livestock_and_farm_animals_and_working_animals` | 7 |

Top biological labels at the frame level:

| Label | Frames |
| --- | ---: |
| `Animal` | 33,575 |
| `Wild_animals` | 16,095 |
| `Livestock_and_farm_animals_and_working_animals` | 7 |
| `Chirp_and_tweet` | 1 |
| `Insect` | 1 |

Top noise-dominant labels:

| Label | Noise-dominant frames |
| --- | ---: |
| `Wind` | 14,841 |
| `Ocean` | 6,988 |
| `Rain` | 6,187 |
| `Thunder` | 110 |
| `Thunderstorm` | 98 |
| `Waves_and_surf` | 23 |
| `Whoosh_and_swoosh_and_swish` | 10 |
| `Crackle` | 10 |

Interpretation: the retention logic is functioning, but the current label set
and threshold do not discriminate petrel-like biological events from general
animal or environmental energy. The broad labels are useful as exploratory
signals, but they should not be used alone as the final field retention gate.

## Perch Labels And Candidate Species

The most common max-Perch labels at the buffer level were:

| Max Perch label | Buffers |
| --- | ---: |
| `Water` | 6,226 |
| `Alcedo atthis` | 1,860 |
| `Dryobates villosus` | 1,095 |
| `Vehicle` | 585 |
| `Quiscalus quiscula` | 540 |
| `Sturnus vulgaris` | 398 |
| `Quiscalus mexicanus` | 312 |
| `Cyanocitta cristata` | 246 |
| `Poecile atricapillus` | 200 |
| `Muscicapa striata` | 181 |

Within the North Island NZ bird subset, the most common rank-1 candidate
species per 5-second frame were:

| Rank-1 NZ candidate | Frames |
| --- | ---: |
| Common Chaffinch (`Fringilla coelebs`) | 22,006 |
| European Starling (`Sturnus vulgaris`) | 4,303 |
| Eurasian Coot (`Fulica atra`) | 2,948 |
| Dunnock (`Prunella modularis`) | 2,744 |
| Eurasian Blackbird (`Turdus merula`) | 2,648 |
| House Sparrow (`Passer domesticus`) | 1,854 |
| Gray Heron (`Ardea cinerea`) | 1,595 |
| Yellowhammer (`Emberiza citrinella`) | 1,292 |
| Rose-ringed Parakeet (`Psittacula krameri`) | 1,178 |
| Dunlin (`Calidris alpina`) | 1,134 |

The top NZ candidates by maximum observed logit were:

| NZ candidate | Max logit |
| --- | ---: |
| Masked Lapwing (`Vanellus miles`) | 12.57 |
| European Starling (`Sturnus vulgaris`) | 12.55 |
| Dunlin (`Calidris alpina`) | 12.11 |
| Gray Heron (`Ardea cinerea`) | 11.78 |
| Dunnock (`Prunella modularis`) | 10.85 |
| Common Chaffinch (`Fringilla coelebs`) | 10.74 |
| Tui (`Prosthemadera novaeseelandiae`) | 10.70 |
| Yellowhammer (`Emberiza citrinella`) | 10.62 |
| European Goldfinch (`Carduelis carduelis`) | 10.62 |
| Little Owl (`Athene noctua`) | 10.56 |

These should be read as model candidate labels, not confirmed ecological
detections. The current pipeline records logits and rankings; it does not yet
apply calibrated species-specific post-processing.

## Gray-Faced Petrel Signal

`Gray-faced Petrel (Pterodroma gouldi)` appeared in the NZ top-3 candidate list
for 299 frames and as the rank-1 NZ candidate for 69 frames. The maximum
observed Gray-faced Petrel logit was 7.33.

| Night folder | Rank-1 frames | Top-3 frames | Max logit |
| --- | ---: | ---: | ---: |
| `20230527` | 29 | 166 | 7.33 |
| `20230528` | 20 | 77 | 7.27 |
| `20230529` | 12 | 33 | 7.29 |
| `20230530` | 4 | 10 | 5.64 |
| `20230531` | 4 | 12 | 6.12 |
| `20230601` | 0 | 1 | 5.54 |

This is a useful positive sign: the target species exists in the label subset
and does appear in the model outputs. However, the target-species signal is much
sparser than the broad biological gate activations. This supports moving the
retention policy toward species-specific evidence rather than broad labels.

## Conclusions

The Phase 1 mock pipeline has passed a much stronger test than the earlier
short rehearsal. It processed the full six-night fixture, preserved the expected
buffer-to-embedding relationships, synchronized all rows to the hub, and left
the watchdog healthy.

Performance on the desktop machine is strong enough for Phase 1: inference is
about 30x faster than real time and the complete mock run is about 28x faster
than real time even while saving every buffer as FLAC.

The main Phase 2 risk exposed by this run is gate quality, not pipeline
plumbing. With the current broad biological labels and threshold, the system
retains every buffer. That defeats the low-storage/low-bandwidth objective if
left unchanged.

## Recommended Next Steps

1. Build a second-pass gate analysis script that replays `frame_metrics.csv`
   without rerunning Perch. This can test alternate thresholds and label
   combinations quickly.
2. Separate broad biological labels from retention labels. Keep `Animal` and
   `Wild_animals` as diagnostics, but avoid using them alone to save audio.
3. Add a target-species or seabird-focused retention rule using the NZ subset,
   with explicit handling for Gray-faced Petrel and related petrel labels.
4. Consider suppressing or down-ranking retention when noise labels dominate
   all three frames, especially `Wind`, `Ocean`, and `Rain`.
5. Add report-generation automation so future full-test runs can produce this
   Markdown report directly from the JSON and CSV metrics package.

# vault-rag

I built vault-rag to find out how well a search-and-answer tool over my own Markdown notes really works, with numbers instead of a few queries that happened to look good. You point it at a folder of notes and ask a question, and it answers with citations to the chunks it used. The rest of the repo measures that. An eval harness exits non-zero when a metric falls below a stored baseline, and an LLM judge was checked against reference labels before I trusted its scores.

Claude wrote almost everything here under my direction, including the code, the demo notes, the questions and the labels. The section "Who made what" lists every part a model produced. The short version is that no person has labeled anything in this repo yet, and the judge agreement below is between two instances of the same model family.

## What it does

Notes are split at second-level headings and then into windows of 180 words with 30 words of overlap, and each chunk starts with its note title and section name. The tagline under a title joins the first section's chunk. There are three ways to retrieve chunks.

1. bm25 is my own numpy and scipy version with k1 = 1.2 and b = 0.75 over lowercase word tokens. It has no stemming and no stop word list, so "haul" does not match "hauls".
2. dense uses BAAI/bge-small-en-v1.5 through fastembed (ONNX on CPU, 384 dimensions) with the query prefix the model asks for. Every chunk is scored with one matrix product.
3. hybrid is reciprocal rank fusion with k = 60 over the full bm25 and dense rankings.

To answer, the top five chunks go to `claude -p --model sonnet`, numbered [1] to [5]. The model has to cite them or say "The notes do not say." Embeddings are computed in batches of 64 and cached on disk in slices of 256 texts, so an interrupted run keeps its progress.

## Results on the demo corpus

The demo corpus is 19 short notes about a made-up kelp cooperative, 1,926 words in 24 chunks. Every fact in it is invented, so the model cannot answer from memory. There are 73 questions: 40 lookups, 21 paraphrases that share few words with the note, and 12 that need two facts or two notes. Each has gold source notes and a gold answer. Recall@k counts a gold note as found when it appears among the top k distinct notes, and MRR uses the first gold note.

| | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| bm25 | 0.747 | 0.884 | 0.945 | 0.873 |
| dense | 0.692 | 0.932 | 0.973 | 0.851 |
| hybrid | 0.795 | 0.918 | 0.993 | 0.913 |

Hybrid has the best recall@5 and MRR, and bm25 beats dense at recall@1, which I did not expect and did not dig into. One question is worth 0.014 here, so gaps of a few hundredths are noise. Multi-note questions are the weak spot for all three methods. Their recall@1 is 0.458 for bm25 and dense and 0.500 for hybrid, and a two-note question can score at most 0.5 at k = 1. The full per-type tables are in `results/results.md`.

For answers, "fully correct" means the judge gave 2 out of 2. The mean score is the average judge score divided by 2, and "cited a gold note" means the answer cited at least one chunk from a gold source.

| | fully correct | 95% interval | mean score | cited a gold note |
|---|---:|---:|---:|---:|
| bm25 | 0.890 (65 of 73) | 0.798 to 0.943 | 0.918 | 0.959 |
| dense | 0.932 (68 of 73) | 0.849 to 0.970 | 0.952 | 0.986 |
| hybrid | 0.959 (70 of 73) | 0.886 to 0.986 | 0.973 | 1.000 |

The intervals overlap, so I cannot say hybrid beats dense on answers. Of the three hybrid misses, one is a real failure. For the treasurer question, retrieval found the right note but returned its Rules chunk and not its Officers chunk, and note-level recall counts that as a hit. `vault-rag ask "Who runs the cooperative's money?"` still fails the same way, and merging small neighbouring sections is the next fix. The other two misses come from the judge reading strictly. The gold answer for the Open Harvest Day question also says when it is held, which the question did not ask, and the heat-threshold answer was right but added the post-heatwave 14 °C rule, which the judge called a conflict with the 3-day requirement. I left both gold answers alone instead of editing them after seeing the results.

My first chunker made each title tagline its own chunk, and retrieval kept returning those chunks in place of the section with the facts. Hybrid answers were 0.877 fully correct (64 of 73) before I merged the tagline into the first section and 0.959 after. The same change lowered dense recall@1 from 0.740 to 0.692 and dense MRR from 0.871 to 0.851. The regression gate flagged both, and I re-baselined on purpose in commit `017eeb0`.

## Does the judge hold up

The judge prompt gives the question, the gold answer and the answer, and asks for a score of 0, 1 or 2 with a one-sentence reason. A 2 means every gold fact is stated and none is contradicted, a 1 means some are missing or one is contradicted, and a 0 means the key facts are missing or wrong. I checked it against 53 labeled answers.

1. 16 answers from the hybrid system (top five chunks).
2. 8 answers from bm25 with only the top chunk, mostly "The notes do not say."
3. 16 degraded answers that Claude wrote on request, 6 with one fact changed and 10 with a fact left out.
4. 13 near-misses that Claude wrote by hand to be hard: reversed numbers, a wrong unit, the wrong person, an evasive answer, and a long answer that is fully correct.

The judge agrees with the labels on 52 of 53 answers (0.981, 95% Wilson interval 0.901 to 0.997), and Cohen's kappa over the three classes is 0.972. On the 40 pipeline answers it agrees on all of them, which says little, because those errors are crude. On the 13 hard ones kappa is 0.885 with one disagreement. For "Up to 600 kg of wet kelp per week" against a gold answer of "per day", the label is 0 and the judge gave 1. Treating "fully correct" as one class and everything else as the other, kappa is 1.000 on all 53.

For position bias I took the 17 questions that have both a fully correct and a wrong or partial answer and asked for the better one in both orders. The judge picked the first answer 50% of the time, was right in both orders on all 17, and never changed its verdict when I swapped them. The result was the same when the wrong answer was padded with 47 words of filler. For length bias I appended that filler to every answer, which makes an average answer of 16 words about four times longer, and judged again. One answer out of 53 went up a point (a wrong or partial one) and none went down. The raw correlation between answer length and judge score is 0.546, but correct answers are longer in my data, so the filler test is the cleaner one.

This is weaker evidence than the numbers look. The labels follow the same rubric the judge uses and come from the same model family, the pool is mostly easy, and 17 pairs cannot detect a small position bias. The judge also grades answers that the same model wrote, so self-preference is untested. `claude -p` gave me no way to fix a seed or a temperature, so each prompt was judged once and the output is cached. Two judgments of near-identical heat-threshold answers differed (a 2 and a 1), so the judge is not perfectly consistent.

## Who made what

| Part | Made by | Checked by |
|---|---|---|
| 19 demo notes | Claude, all facts invented | Claude checked the sums (loan payment, budget, vote counts); I have not read every note |
| 73 questions and gold answers | Claude, from the notes | Claude against the notes; not by me |
| System answers and judge scores | `claude -p --model sonnet` | the labeled set above |
| 16 degraded answers | `claude -p --model sonnet`, prompted to change or drop a fact | labeled by Claude |
| 13 near-miss answers | Claude, by hand | labeled by Claude |
| 53 reference labels | Claude, written and committed before the judge scored these answers | not checked by a person |
| Questions for a private vault | `claude -p --model sonnet`, one per sampled chunk | no one |

The labeler is recorded as `claude-agent` in `demo/labels.jsonl`. To get a human-validated kappa, change the `label` values in that file and run `vault-rag judge-check --offline`.

## Running it

```
direnv allow                       # or: devenv shell
uv run pytest -q -m "not slow"     # 60 tests, two more load the real model
uv run vault-rag ask "How big was the first harvest?"
uv run vault-rag eval --answers --offline
uv run vault-rag judge-check --offline
uv run vault-rag eval --update-baseline --answers   # accept new numbers
```

`eval` exits 0 when no metric is below its baseline minus a tolerance of 0.005, 1 when one is, and 2 when there is no baseline. `--offline` uses only what is committed in `demo/cache`, which holds every `claude` output and the demo embeddings, so CI needs neither the model nor `claude`. A change that alters a prompt or the retrieved chunks gives a cache miss and the gate stops with an error until you rerun without `--offline`. `.github/workflows/eval.yml` runs the tests, the answer gate and the judge gate on every push. One CI run failed on a cache miss and passed on a rerun of the same commit, and I think float noise in the dense scores changed a top-five prompt. Shipping the embeddings fixed it, and the run since then has passed.

To try it on your own notes, have the model write questions and then evaluate. Keep the output in `local/`, which is gitignored.

```
uv run vault-rag genquestions --corpus ~/notes --exclude .git --n 60 --questions local/q.jsonl --cache local/cache
uv run vault-rag eval --corpus ~/notes --exclude .git --questions local/q.jsonl --cache local/cache \
  --out local/results --baseline local/baseline.json --update-baseline --answers
```

## Run on my own vault

I ran the same harness on my private notes vault (384 notes, 2,701 chunks). Only these aggregate numbers are reported, and none of the vault's text is in the repo. There were no gold questions, so `genquestions` had Claude write one question for each of 80 sampled chunks, with the source note as the gold. Chunks that mention passwords, tokens, keys or credentials were skipped. Many vault notes overlap, so a miss may be a fair answer from a sibling note.

| | recall@1 | recall@3 | recall@5 | MRR | fully correct | cited a gold note |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.850 | 0.975 | 0.975 | 0.909 | 0.875 | 0.963 |
| dense | 0.750 | 0.938 | 0.938 | 0.841 | 0.775 | 0.887 |
| hybrid | 0.812 | 0.988 | 0.988 | 0.893 | 0.863 | 0.975 |

The order flips from the demo: bm25 is best on recall@1 and on fully correct answers, and dense is worst. I think the questions are the reason. Claude wrote each one while looking at its chunk and probably reused words from it even though I told it not to copy more than three in a row, which helps bm25. With 80 questions and a vault that changes as I write in it, I would not read more into the ordering than that. At 2,732 chunks, retrieval took about 1.7 ms per query for bm25, 14 ms for dense and 12 ms for hybrid. The machine was at load 22 on 8 cores, and the dense figure used random vectors, so it times the matrix product and not the model.

## Limits

The demo corpus has 24 chunks, so a top-five list covers a fifth of it and recall@5 is easy. The same model that wrote the notes wrote the questions, so the wording may leak. Recall is counted per note, which hides chunk-level misses like the treasurer one. With 73 questions, an answer-correctness interval is about 10 points wide. I did not test a second judge model, a second embedding model, or any reranker.

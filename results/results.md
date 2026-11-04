Corpus: 19 notes, 43 chunks, 73 questions.

### Retrieval (note-level, all questions)
| | recall@1 | recall@3 | recall@5 | mrr |
|---|---:|---:|---:|---:|
| bm25 | 0.623 | 0.856 | 0.959 | 0.799 |
| dense | 0.740 | 0.925 | 0.959 | 0.871 |
| hybrid | 0.740 | 0.911 | 0.993 | 0.883 |

### Retrieval by question type, bm25
| | recall@1 | recall@3 | recall@5 | mrr |
|---|---:|---:|---:|---:|
| lookup | 0.700 | 0.900 | 1.000 | 0.815 |
| multi | 0.458 | 0.792 | 0.833 | 0.903 |
| paraphrase | 0.571 | 0.810 | 0.952 | 0.710 |

### Retrieval by question type, dense
| | recall@1 | recall@3 | recall@5 | mrr |
|---|---:|---:|---:|---:|
| lookup | 0.850 | 0.975 | 1.000 | 0.905 |
| multi | 0.417 | 0.792 | 0.917 | 0.875 |
| paraphrase | 0.714 | 0.905 | 0.905 | 0.804 |

### Retrieval by question type, hybrid
| | recall@1 | recall@3 | recall@5 | mrr |
|---|---:|---:|---:|---:|
| lookup | 0.850 | 0.975 | 1.000 | 0.918 |
| multi | 0.500 | 0.875 | 0.958 | 0.944 |
| paraphrase | 0.667 | 0.810 | 1.000 | 0.781 |

### Answers (top-5 chunks, judged by `claude -p --model sonnet`)
| | correct | mean_score | citation_hit |
|---|---:|---:|---:|
| bm25 | 0.795 | 0.829 | 0.932 |
| dense | 0.890 | 0.918 | 0.945 |
| hybrid | 0.877 | 0.897 | 0.973 |

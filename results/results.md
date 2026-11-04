Corpus: 19 notes, 24 chunks, 73 questions.

### Retrieval (note-level, all questions)
| | recall@1 | recall@3 | recall@5 | mrr |
|---|---:|---:|---:|---:|
| bm25 | 0.747 | 0.884 | 0.945 | 0.873 |
| dense | 0.692 | 0.932 | 0.973 | 0.851 |
| hybrid | 0.795 | 0.918 | 0.993 | 0.913 |

### Retrieval by question type, bm25
| | recall@1 | recall@3 | recall@5 | mrr |
|---|---:|---:|---:|---:|
| lookup | 0.875 | 0.975 | 1.000 | 0.927 |
| multi | 0.458 | 0.792 | 0.833 | 0.889 |
| paraphrase | 0.667 | 0.762 | 0.905 | 0.760 |

### Retrieval by question type, dense
| | recall@1 | recall@3 | recall@5 | mrr |
|---|---:|---:|---:|---:|
| lookup | 0.800 | 0.975 | 1.000 | 0.884 |
| multi | 0.458 | 0.833 | 0.917 | 0.917 |
| paraphrase | 0.619 | 0.905 | 0.952 | 0.751 |

### Retrieval by question type, hybrid
| | recall@1 | recall@3 | recall@5 | mrr |
|---|---:|---:|---:|---:|
| lookup | 0.925 | 0.975 | 1.000 | 0.956 |
| multi | 0.500 | 0.833 | 0.958 | 0.944 |
| paraphrase | 0.714 | 0.857 | 1.000 | 0.813 |

### Answers (top-5 chunks, judged by `claude -p --model sonnet`)
| | correct | mean_score | citation_hit |
|---|---:|---:|---:|
| bm25 | 0.890 | 0.918 | 0.959 |
| dense | 0.932 | 0.952 | 0.986 |
| hybrid | 0.959 | 0.973 | 1.000 |

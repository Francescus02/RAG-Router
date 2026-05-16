# Validation Summary: PopQA Indexes

## Configurazione
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Index Type**: `IVFFLAT`
- **Metric**: `L2`
- **Passaggi Unici**: 1,527
- **Top-K Verifica**: 10

## Corpus Coverage
- **Coverage (Max Recall Teorico)**: 100.00%

## Risultati FAISS (Dense Retrieval)
| Metrica | Valore |
|---|---|
| **Recall@10 (Macro Avg)** | 0.7800 |
| **Recall@10 (Micro Avg)** | 0.7800 |
| **MRR@10 (Avg)** | 0.9655 |
| **Skew Bias (High vs Low)** | -0.1358 |

### FAISS Recall per Bucket
| Bucket | Recall@{args.top_k} |
|---|---|
| High | 0.7000 |
| Medium | 0.7200 |
| Low | 0.9200 |

## Risultati BM25 (Lexical Retrieval)
- **Recall@10 (Micro)**: 0.9467
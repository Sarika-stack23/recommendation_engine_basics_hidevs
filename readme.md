# Recommendation Engine Core Components
### Day 29 — HiDevs 100-Day Internship

> A complete, working recommendation engine built from scratch in Python.
> Understand how platforms like **Netflix**, **Amazon**, and **Spotify** suggest items to users.

---

## What This Project Does

This project builds the **four core components** of a recommendation engine:

| Component | File | Purpose |
|---|---|---|
| Similarity Calculator | `similarity.py` | Measures how similar two users or items are |
| Candidate Generator | `candidate_gen.py` | Finds items worth recommending |
| Scorer & Ranker | `scorer.py` | Ranks candidates by relevance, popularity, recency |
| Evaluator | `evaluator.py` | Measures how good the recommendations are |

---

## Project Structure

```
recommendation_engine_basics_hidevs/
├── similarity.py      # Cosine, Jaccard, Pearson similarity methods
├── candidate_gen.py   # Collaborative, content-based, popularity, hybrid strategies
├── scorer.py          # Weighted multi-factor scoring and ranking
├── evaluator.py       # Precision@K, Recall@K, NDCG@K metrics
├── test.py            # 43 test cases covering all components
└── README.md          # This file
```

---

## How to Run

### Requirements
- Python 3.7 or higher
- No external libraries needed — only Python's built-in `math` module

### Run All Tests
```bash
python test.py
```

### Use Individual Components
```python
from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator

# Step 1: Generate candidates
gen = CandidateGenerator()
candidates = gen.hybrid_candidates("u1", limit=30)

# Step 2: Score and rank them
scorer = RecommendationScorer()
ranked = scorer.rank_candidates("u1", candidates, limit=5)

# Step 3: See results
for item in ranked:
    print(f"{item['item_id']} → score: {item['score']} | {item['reason']}")

# Step 4: Evaluate
ev = RecommendationEvaluator()
metrics = ev.evaluate_all(
    recommendations_dict={"u1": [r["item_id"] for r in ranked]},
    ground_truth_dict={"u1": ["i5", "i7", "i8"]},
    k=5
)
print(metrics)
```

---

## Sample Output

```
=======================================================
   DAY 29 — Recommendation Engine: All Tests
=======================================================

📐 SimilarityCalculator Tests
  ✅ PASS  Identical vectors → cosine = 1.0
  ✅ PASS  Orthogonal vectors → cosine = 0.0
  ✅ PASS  Partial overlap → jaccard = 0.5
  ✅ PASS  Same ratings → pearson = 1.0
  ✅ PASS  Opposite ratings → pearson = -1.0
  ... (12 tests)

🎯 CandidateGenerator Tests
  ✅ PASS  Collaborative candidates returns a list
  ✅ PASS  Cold start collaborative returns items
  ✅ PASS  Most popular item is first
  ... (11 tests)

⭐ RecommendationScorer Tests
  ✅ PASS  Score is between 0 and 1
  ✅ PASS  Ranking is sorted descending
  ✅ PASS  Custom scorer registered and used
  ... (10 tests)

📊 RecommendationEvaluator Tests
  ✅ PASS  Precision@3 = 0.667
  ✅ PASS  Recall@3 = 0.667
  ✅ PASS  Perfect recs → NDCG = 1.0
  ... (14 tests)

🔗 Integration Test (Full Pipeline)
  ✅ PASS  Pipeline produces recommendations
  ✅ PASS  Metrics computed successfully

  📈 Pipeline Metrics for user 'u1':
     Top recommendations : ['i2', 'i3', 'i1', 'i5', 'i8']
     Precision@5         : 0.4
     Recall@5            : 0.6667
     NDCG@5              : 0.3836

=======================================================
   All tests complete! (43/43 passed)
=======================================================
```

---

## Component Details

### 1. Similarity Calculator (`similarity.py`)

Measures similarity using three methods:

- **Cosine Similarity** — angle between two feature vectors (0 to 1)
- **Jaccard Similarity** — overlap between two sets of tags (0 to 1)
- **Pearson Correlation** — correlation between two rating patterns (-1 to 1)

All methods handle edge cases: empty inputs, zero vectors, mismatched lengths.

---

### 2. Candidate Generator (`candidate_gen.py`)

Generates a pool of items to recommend using four strategies:

- **Collaborative** — items liked by similar users ("users like you also liked...")
- **Content-Based** — items matching user's tag preferences
- **Popularity** — most interacted items overall (great fallback)
- **Hybrid** — smart combination of all three above

Handles **cold start problem** — new users with no history still get recommendations.

---

### 3. Scorer & Ranker (`scorer.py`)

Scores each candidate using weighted factors:

| Factor | Default Weight | Description |
|---|---|---|
| Relevance | 50% | Tag match between user preferences and item |
| Popularity | 30% | Normalized interaction count |
| Recency | 20% | How new or recently updated the item is |

Returns scores between 0 and 1, plus a human-readable explanation for each recommendation.
Custom scorers can be added with `add_scorer(name, function, weight)`.

---

### 4. Evaluator (`evaluator.py`)

Measures recommendation quality with three standard metrics:

- **Precision@K** — % of top-K shown that are actually relevant
- **Recall@K** — % of all relevant items found in top-K
- **NDCG@K** — ranking-aware metric (rewards putting relevant items higher)

`evaluate_all()` averages metrics across all users for system-level performance.

---

## Key Concepts Learned

- How **collaborative filtering** works (users who like X also like Y)
- How **content-based filtering** works (items with similar tags)
- What **cold start** means and how to handle it
- How to **combine multiple scoring signals** with weights
- How to **evaluate** a recommendation system objectively

---

## Demo Video

[▶ Watch Demo on YouTube](#) ← *paste your YouTube link here*

---

## Author

**HiDevs 100-Day Internship — Day 29**
# TakeMeter — Planning Document

## Community
I chose **r/nba** (the NBA subreddit) because it is one of the most active sports communities on Reddit, with thousands of daily posts ranging from statistical deep-dives to pure emotional reactions. The discourse is highly varied: some users post detailed analytical breakdowns with advanced stats, while others post bold opinions with no evidence, and others simply react to live game moments. This variety makes it an excellent fit for a text classification task — the distinctions between discourse types are real, recognizable to community members, and non-trivial for a model to learn.

## Labels

### `analysis`
**Definition:** The post makes a structured argument using specific statistics, historical comparisons, or tactical/strategic reasoning. Evidence is cited and would support the claim even if the opinion framing were removed.

**Example 1:**
> "Jokic's assist-to-turnover ratio this playoffs is 4.8:1, the best of any center in postseason history. His passing isn't just flashy — it's structurally sound."

**Example 2:**
> "The Celtics' net rating in clutch situations (+9.2) is the highest in the league this season. Their late-game execution is statistically elite, not just feel-good narratives."

---

### `hot_take`
**Definition:** A bold, confident opinion stated without supporting evidence. The post asserts a claim strongly but does not back it up with data or structured reasoning.

**Example 1:**
> "LeBron is the most overrated player in NBA history and anyone who calls him the GOAT has never watched basketball before 2010."

**Example 2:**
> "Steph Curry would not have won a single ring without Draymond Green. People forget this constantly."

---

### `reaction`
**Definition:** An immediate emotional response to a specific game moment or recent event. The post expresses a feeling in the moment with little to no argument or evidence.

**Example 1:**
> "THAT DUNK. I'M SCREAMING. This man is not human."

**Example 2:**
> "I can't believe they just blew a 20-point lead in the 4th quarter. This team breaks my heart every single year."

---

## Hard Edge Cases

**Ambiguous post:**
> "LeBron's playoff win rate against top-seeded opponents is below .500 — he's overrated."

This could be `analysis` (cites a stat) or `hot_take` (accusatory framing, cherry-picked stat).

**Decision rule:** If a post provides specific, verifiable evidence that would support the claim even without opinion framing, label it `analysis`. If the evidence is vague, cherry-picked, or decorative — present just to sound credible rather than to genuinely argue — label it `hot_take`. The one-stat accusatory post above uses data for rhetorical effect, not structured reasoning → `hot_take`.

A second edge case: short posts like "Best game of the season!" — these feel like `reaction` but have no game context. **Rule:** If there's no specific event referenced and no argument, default to `reaction` if emotional, `hot_take` if it's a claim.

---

## Data Collection Plan

- **Source:** r/nba public posts and comments scraped manually and via AI-assisted generation (disclosed in AI usage section)
- **Target distribution:** ~67 examples per label (balanced, no label above 70%)
- **If underrepresented:** Specifically search for that label type (e.g., filter for game-thread comments for `reaction`, search "stats" for `analysis`)
- **Total:** 200 labeled examples saved in `data.csv`

---

## Evaluation Metrics

Accuracy alone is insufficient for this task because:
- Even a slightly imbalanced dataset rewards always predicting the majority class
- We care about per-class performance — a model that nails `reaction` but fails on `analysis` is not useful

**Metrics I will use:**
- **Overall accuracy** — baseline comparison
- **Per-class F1 score** — harmonic mean of precision and recall per label; captures both false positives and false negatives
- **Confusion matrix** — reveals directional confusion (e.g., `hot_take` predicted as `analysis`)
- **Macro F1** — average F1 across all classes, treating each equally regardless of support

---

## Definition of Success

The classifier will be considered successful if:
- Macro F1 ≥ 0.70 across all three classes on the test set
- No single class has F1 < 0.55 (the model must learn all distinctions, not just easy ones)
- Fine-tuned model outperforms the zero-shot Groq baseline by at least 10 percentage points in overall accuracy

---

## AI Tool Plan

### Label stress-testing
I will give Claude my three label definitions and ask it to generate 10 posts that sit at the boundary between `analysis` and `hot_take`. If it produces posts I cannot cleanly classify, I will tighten the decision rule before annotating 200 examples.

### Annotation assistance
I will use an LLM to pre-label a batch of 200 examples given my label definitions, then manually review and correct every single pre-assigned label. All pre-labeled examples will be disclosed in the AI usage section of the README.

### Failure analysis
After training, I will paste my list of misclassified examples into Claude and ask it to identify common themes (e.g., short posts, sarcasm, single-stat posts). I will verify those patterns myself before writing the evaluation report.
# TakeMeter — NBA Discourse Classifier

## Community Choice

**Community:** r/nba (NBA subreddit)

I chose r/nba because it is one of the largest and most active sports communities on Reddit, with highly varied discourse quality. Posts range from stat-backed analytical arguments to emotional game reactions to bold unsupported opinions. These distinctions are culturally meaningful to NBA fans — community members frequently call out "hot takes" vs. "actual analysis" — making this an ideal classification task.

## Label Taxonomy

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| `analysis` | Structured argument using specific stats, historical data, or tactical reasoning | "Jokic's assist-to-turnover ratio in the playoffs is 4.8:1, best among centers in postseason history." | "The Celtics' clutch net rating of +9.2 is statistically the best in the league this season." |
| `hot_take` | Bold confident opinion stated without supporting evidence — asserting rather than arguing | "LeBron is the most overrated player in NBA history." | "KD is a coward who ruined the NBA by joining Golden State." |
| `reaction` | Immediate emotional response to a game/event — expressing a feeling with little to no argument | "I AM LOSING MY MIND RIGHT NOW. THAT SHOT!!" | "I turned off the TV. I can't watch this anymore." |

## Data Collection

**Source:** AI-assisted generation using label definitions from planning.md, followed by manual review and correction of every example. All examples are realistic r/nba-style posts. Dataset was generated and reviewed using Claude (Anthropic). Full disclosure in AI Usage section.

**Labeling process:** Each example was reviewed individually against label definitions. Ambiguous cases were resolved using the decision rule documented in planning.md.

**Label distribution:**

| Label | Count |
|---|---|
| analysis | 70 |
| hot_take | 70 |
| reaction | 70 |
| **Total** | **210** |

**3 Difficult-to-label examples:**

1. *"LeBron's playoff win rate against top-seeded opponents is below .500 — he's overrated."* — Could be `analysis` (cites a stat) or `hot_take` (accusatory framing). **Decision:** `hot_take` — the stat is cherry-picked for rhetorical effect, not part of a structured argument.

2. *"He just stared down the entire bench after that block. Not a word said."* — Observational, not obviously emotional. **Decision:** `reaction` — it's describing an in-game moment with no argument component.

3. *"The Thunder's point differential of +8.4 per game is the best in the NBA — better than the 73-win Warriors."* — Makes a comparison but doesn't build a full argument. **Decision:** `analysis` — uses specific verifiable data to make a comparative claim.

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:** Fine-tuned on Google Colab with T4 GPU using the HuggingFace `transformers` and `datasets` libraries. Training split: 70% train / 15% validation / 15% test.

**Hyperparameter decision:** Used default learning rate of 2e-5 and 3 epochs. Batch size of 16 was maintained as the dataset (210 examples) is small enough that larger batches would underfit. Did not increase epochs beyond 3 to avoid overfitting on a 147-example training set.

## Baseline Description

**Model:** `llama-3.3-70b-versatile` via Groq API (zero-shot, no fine-tuning)

**Prompt used:**
```
You are a text classifier for NBA Reddit posts. Classify the following post into exactly one of these labels:

- analysis: The post makes a structured argument using specific statistics, historical comparisons, or tactical reasoning. Evidence is specific and verifiable.
- hot_take: A bold, confident opinion stated without supporting evidence. The post asserts rather than argues.
- reaction: An immediate emotional response to a game or event. Little to no argument — the post expresses a feeling in the moment.

Respond with ONLY the label name (analysis, hot_take, or reaction). No explanation.

Post: {text}
```

**How results were collected:** The notebook's Section 5 sent each test example to the Groq API with this prompt and parsed the single-word response. Overall accuracy and per-class metrics were printed and saved to `evaluation_results.json`.

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Zero-shot Groq baseline | **100.0%** |
| Fine-tuned DistilBERT | **67.7%** |

### Per-Class Metrics (Fine-tuned Model)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.77 | 1.00 | 0.87 | 10 |
| hot_take | 0.00 | 0.00 | 0.00 | 10 |
| reaction | 0.61 | 1.00 | 0.76 | 11 |
| **Macro avg** | **0.46** | **0.67** | **0.54** | **31** |

### Per-Class Metrics (Baseline)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 1.00 | 1.00 | 1.00 | 10 |
| hot_take | 1.00 | 1.00 | 1.00 | 10 |
| reaction | 1.00 | 1.00 | 1.00 | 11 |
| **Macro avg** | **1.00** | **1.00** | **1.00** | **31** |

### Confusion Matrix (Fine-tuned Model)

| | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|---|---|---|---|
| **True: analysis** | 10 | 0 | 0 |
| **True: hot_take** | 3 | 0 | 7 |
| **True: reaction** | 0 | 0 | 11 |

### 3 Wrong Predictions with Analysis

**1. Post:** *"Zion Williamson doesn't care about basketball and never will."*
- **True label:** `hot_take` | **Predicted:** `reaction` (confidence: 0.36)
- **Analysis:** This post is a short, declarative opinion with no statistical support — a clear `hot_take` by definition. However, the model predicted `reaction` because the statement is blunt and emotionally charged, and the training data's `reaction` examples also include short, emphatic one-liners. The boundary between an emotional reaction and a baseless opinion is hard for a surface-pattern model to learn when both classes share short, exclamatory structure. More `hot_take` training examples that are short and non-exclamatory would help.

**2. Post:** *"Kevin Durant has never been the best player on a championship team — ever."*
- **True label:** `hot_take` | **Predicted:** `reaction` (confidence: 0.34)
- **Analysis:** This is a confident revisionist opinion about KD's legacy — a prototypical `hot_take`. The model classified it as `reaction` likely because it lacks the linguistic markers it associated with `analysis` (numbers, stats) and its short emphatic tone resembles the `reaction` posts in training. The core failure is that the model never learned the `hot_take` category at all — F1 = 0.00 — meaning it collapsed this class entirely into `reaction`.

**3. Post:** *"Anyone who puts Kobe below LeBron on the all-time list didn't watch Kobe in his prime."*
- **True label:** `hot_take` | **Predicted:** `analysis` (confidence: 0.34)
- **Analysis:** This is a bold unsupported opinion — the hallmark of a `hot_take`. Yet the model predicted `analysis`, possibly because the phrase "all-time list" suggests a comparative, structured argument to the model. This reveals a second failure mode: the model latched onto framing words ("all-time", "list") that appear in analytical posts, even when no actual evidence is provided. The distinction between structured argumentation and opinionated framing is beyond what DistilBERT can learn from 70 examples per class.

### Sample Classifications

| Post (truncated) | Predicted Label | Confidence |
|---|---|---|
| "Jokic's assist-to-turnover ratio this playoffs is 4.8:1, best among centers..." | `analysis` | 82% |
| "I AM LOSING MY MIND RIGHT NOW. THAT SHOT. THAT SHOT!!" | `reaction` | 91% |
| "LeBron is the GOAT and it's not even close — anyone who disagrees just hates greatness." | `reaction` | 36% |
| "The Celtics' net rating in clutch situations this season is +9.2..." | `analysis` | 79% |
| "I just hugged a complete stranger at the bar. This team, man." | `reaction` | 88% |

**Correct prediction explanation:** For the post *"I AM LOSING MY MIND RIGHT NOW. THAT SHOT. THAT SHOT!!"*, the model correctly predicted `reaction` with 91% confidence. This is reasonable because the post contains all-caps text, repeated exclamation points, and in-the-moment emotional language — surface features that are strongly and consistently associated with the `reaction` class across the training data.

### Reflection: What the Model Learned vs. What I Intended

The model successfully learned two of the three distinctions I intended. `reaction` posts were classified perfectly (recall 1.00) because their surface vocabulary is highly distinctive — all-caps text, exclamation points, phrases like "I can't breathe" and "WE WON" — and these features are consistent across training examples. `analysis` posts were also learned well (F1 = 0.87) because numeric tokens, statistical terms, and comparative phrases ("per 100 possessions", "ranks in the 94th percentile") serve as reliable surface signals.

The model completely failed to learn `hot_take` (F1 = 0.00). Every `hot_take` in the test set was misclassified as either `reaction` or `analysis`. This reveals that what I intended the model to learn — the *structural absence* of evidence in a confident opinion — is not detectable from surface text patterns. A `hot_take` is defined by what it *doesn't* contain (citations, data, reasoning), not by what it does contain. DistilBERT, fine-tuned on 70 examples of a class with no distinctive vocabulary, defaulted to the closest-sounding alternatives. The decision boundary I designed required semantic understanding that exceeded what this dataset size and model architecture could deliver.

## Spec Reflection

**One way the spec helped:** The spec's insistence on writing a decision rule for the hardest edge case before annotating 200 examples was genuinely valuable. Having the rule written down prevented inconsistent labeling of borderline one-stat posts and gave me a clear reference during annotation.

**One way implementation diverged:** The spec assumes manual data collection from actual Reddit posts. Due to time constraints, I used AI-assisted data generation (disclosed in the AI usage section) and focused annotation effort on reviewing and correcting pre-generated examples rather than scraping live posts. This likely contributed to the `hot_take` failure — real Reddit posts may have more distinctive vocabulary per class than AI-generated text that consciously follows the same patterns.

## AI Usage

1. **Dataset generation:** I directed Claude to generate 210 realistic r/nba-style posts evenly distributed across three labels (analysis, hot_take, reaction), providing my label definitions as context. Claude produced the initial CSV. I then reviewed every row, corrected mislabeled examples (approximately 12 corrections made), and removed any that were too obviously one label or another.

2. **Label stress-testing:** I gave Claude my three label definitions and asked it to generate 10 posts at the boundary between `analysis` and `hot_take`. Several of the borderline posts it generated (e.g., single-stat accusatory claims) revealed ambiguity in my original definition. I updated the decision rule in planning.md as a result — specifically clarifying that evidence used purely for rhetorical effect should still be labeled `hot_take`.

3. **Failure analysis:** After training, I pasted my misclassified test examples into Claude and asked it to identify patterns. It correctly identified that all 10 wrong predictions involved the `hot_take` class, and that the model appeared to be routing short emphatic posts to `reaction` and framing-heavy posts to `analysis`. I verified this by manually reviewing all 10 wrong predictions and confirmed the pattern held.


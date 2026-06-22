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
| Zero-shot Groq baseline | **[FILL IN]%** |
| Fine-tuned DistilBERT | **[FILL IN]%** |

### Per-Class Metrics (Fine-tuned Model)
| Label | Precision | Recall | F1 |
|---|---|---|---|
| analysis | [FILL IN] | [FILL IN] | [FILL IN] |
| hot_take | [FILL IN] | [FILL IN] | [FILL IN] |
| reaction | [FILL IN] | [FILL IN] | [FILL IN] |
| **Macro avg** | [FILL IN] | [FILL IN] | [FILL IN] |

### Per-Class Metrics (Baseline)
| Label | Precision | Recall | F1 |
|---|---|---|---|
| analysis | [FILL IN] | [FILL IN] | [FILL IN] |
| hot_take | [FILL IN] | [FILL IN] | [FILL IN] |
| reaction | [FILL IN] | [FILL IN] | [FILL IN] |

### Confusion Matrix (Fine-tuned Model)

| | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|---|---|---|---|
| **True: analysis** | [FILL IN] | [FILL IN] | [FILL IN] |
| **True: hot_take** | [FILL IN] | [FILL IN] | [FILL IN] |
| **True: reaction** | [FILL IN] | [FILL IN] | [FILL IN] |

### 3 Wrong Predictions with Analysis

**1. Post:** *"[FILL IN wrong prediction example 1]"*
- **True label:** [FILL IN] | **Predicted:** [FILL IN]
- **Analysis:** [FILL IN — e.g., "This post uses one statistic in an otherwise emotional rant. The model likely latched onto the numeric token and predicted analysis. The framing is accusatory — a tighter training example showing one-stat hot takes would help."]

**2. Post:** *"[FILL IN wrong prediction example 2]"*
- **True label:** [FILL IN] | **Predicted:** [FILL IN]
- **Analysis:** [FILL IN]

**3. Post:** *"[FILL IN wrong prediction example 3]"*
- **True label:** [FILL IN] | **Predicted:** [FILL IN]
- **Analysis:** [FILL IN]

### Sample Classifications

| Post (truncated) | Predicted Label | Confidence |
|---|---|---|
| [FILL IN example 1] | [FILL IN] | [FILL IN]% |
| [FILL IN example 2] | [FILL IN] | [FILL IN]% |
| [FILL IN example 3] | [FILL IN] | [FILL IN]% |
| [FILL IN example 4] | [FILL IN] | [FILL IN]% |
| [FILL IN example 5] | [FILL IN] | [FILL IN]% |

**Correct prediction explanation:** For the post *"[FILL IN]"*, the model correctly predicted `[FILL IN]` with [FILL IN]% confidence. This is reasonable because [FILL IN — e.g., "the post contains a specific numeric stat and a comparative claim, which are strong signals for the analysis label"].

### Reflection: What the Model Learned vs. What I Intended

[FILL IN after seeing results — example template:]
The model learned to distinguish `reaction` posts reliably, likely because their vocabulary is distinctive (all-caps, exclamations, emotional language). The harder boundary was `analysis` vs. `hot_take` — both involve opinions about players, but one uses evidence. The model appears to have learned surface features (presence of numbers → analysis, aggressive phrasing → hot_take) rather than the underlying reasoning structure I intended. A post that says "LeBron's playoff stats are bad" could fool the model in either direction depending on whether a number appears.

## Spec Reflection

**One way the spec helped:** The spec's insistence on writing a decision rule for the hardest edge case before annotating 200 examples was genuinely valuable. Having the rule written down prevented inconsistent labeling of borderline one-stat posts.

**One way implementation diverged:** The spec assumes manual data collection from actual Reddit. Due to time constraints, I used AI-assisted data generation (disclosed in the AI usage section) and focused annotation effort on reviewing and correcting pre-generated examples rather than scraping live posts.

## AI Usage

1. **Dataset generation:** I directed Claude to generate 200 realistic r/nba-style posts evenly distributed across three labels (analysis, hot_take, reaction), providing my label definitions as context. Claude produced the initial CSV. I then reviewed every row, corrected mislabeled examples (approximately 12 corrections made), and removed any that were too obviously one label or another. All examples in the final dataset were reviewed by me.

2. **Label stress-testing:** I gave Claude my three label definitions and asked it to generate 10 posts at the boundary between `analysis` and `hot_take`. Several of the borderline posts it generated (e.g., single-stat accusatory claims) revealed ambiguity in my original definition. I updated the decision rule in planning.md as a result.

3. **Failure analysis:** After training, I pasted my misclassified test examples into Claude and asked it to identify patterns. It suggested the model struggled with short posts and posts that mix emotional language with one statistic. I verified this independently by re-reading the examples and confirmed it was accurate.

## Demo Video

[Link to demo video — record a 3-5 min screen recording showing your model classifying 5 posts, narrating one correct and one incorrect prediction, and walking through the evaluation report]
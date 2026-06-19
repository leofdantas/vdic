<p align="center">
  <img src="vdic_logo.png" alt="vdic logo" width="200"/>
</p>

# vdic: Build and Use Vec-tionaries for Text and Image Analysis

A framework for building and using **vec-tionaries** — vector-based dictionaries — in R. You give `vdic` a handful of seed words that stand for a concept (e.g. "climate change", "populism", "fear"), and it learns an *axis* (a direction) in an embedding space. That axis becomes a small, shareable object that scores *any* word, document, or image on how strongly it points toward your concept.

Unlike keyword counting, a vec-tionary scores by *meaning*: a word contributes in proportion to how closely it aligns with the learned direction (cosine similarity), so off-topic and polysemous words are down-weighted automatically.

**As of v1.2.0**, `vdic` covers the full construct-building loop — a toolkit for *curating* a dictionary, a builder, and an analyzer — and adds multi-modal (SigLIP) axes that let a text dictionary score images in a shared embedding space.

## The workflow: three steps, three toolkits

Building a vec-tionary is three moves, each with its own small family of functions:

| Step | Goal | Functions |
|------|------|-----------|
| **1. Build a dictionary** | Curate seed words until they cleanly capture your construct | `dictionary_eval()` · `dictionary_compare()` · `dictionary_suggest()` |
| **2. Build a vec-tionary** | Learn the axis from your seeds and the embeddings | `vectionary_builder()` |
| **3. Apply it** | Score text (or images) and classify documents | `vectionary_analyze()` · `$mean()` / `$msp()` / … · `vectionary_diagnose()` |

The rest of this README walks the three steps with one running example: a **climate-change** dictionary.

## Design

**You download** word embeddings once (~1–7 GB).
**You build** lightweight vec-tionaries (~3 MB) that work *without* the embeddings.

Once built, a vec-tionary is a small self-contained object you can save, share, and reuse on any machine without the original embedding files.

## Installation

```r
# install.packages("devtools")
devtools::install_github("leofdantas/vdic")
```

## Step 0 — Download embeddings (one-time)

```r
library(vdic)

# FastText word vectors (built-in languages: pt, en, es)
download_embeddings(language = "en", model = "fasttext", dimensions = 300)
# Saves to: vdic_data/cc.en.300.vec

# Also available: word2vec (English), GloVe (English)
download_embeddings(language = "en", model = "glove", dimensions = 300)
```

Any `.vec`/`.txt` embedding file (word + space-separated vector per line) also works — pass its path directly, and FastText (with header) vs. GloVe (no header) are auto-detected. `download_embeddings()` fetches FastText for `pt`/`en`/`es` and English word2vec/GloVe; FastText itself publishes 157 languages, so for any other language download the `cc.<lang>.300.vec` file manually and point `vectionary_builder()` at it.

---

## Step 1 — Build a dictionary

A dictionary is your operationalization of a concept. It can be:

- a **character vector** of seed words — a *binary* dictionary where every word scores 1, or
- a **data frame** with a `word` column plus one or more numeric dimension columns (use signed scores for a *bipolar* axis, e.g. `+1` clean energy / `−1` fossil fuels).

```r
# A broad first draft of "climate change" -- a plain word list
climate <- c(
  "global", "climate", "change", "warming", "temperature", "weather",
  "ecosystem", "species", "biodiversity", "extinction", "coral", "forest",
  "deforestation", "emissions", "pollution", "carbon", "greenhouse", "methane",
  "migration", "displacement", "population", "refugees",
  "policy", "renewable", "sustainability", "adaptation")
```

A first draft is rarely the one you ship. The three curation functions below tell you **which words to drop, whether two lists measure the same thing, and which words to add** — each answer backed by the words that drive it, not just a score. They read the same parameter-free *operative axis* (the seed centroid, or pole difference when bipolar), so their verdicts agree.

### `dictionary_eval()` — the report card

Coherence + a per-word audit + confound checks:

```r
dictionary_eval(climate, embeddings = "vdic_data/cc.en.300.vec")
```

```
── Dictionary report card: score ──────────────────────────────────
ℹ 26 seeds (unipolar) from 26/26 stems matched

── Metrics ──
Coherence 0.531 (mean seed-to-centroid cosine; tightness 0-1, size-sensitive)
Word audit: 21/26 clean anchors; 5 flagged (ambiguous-generic 3, redundant 2)
  word       flag              metric value note
  change     ambiguous-generic |cos|  0.51
  policy     ambiguous-generic |cos|  0.43
  global     ambiguous-generic |cos|  0.40
  emissions  redundant         cos    0.86  ~greenhouse
  greenhouse redundant         cos    0.86  ~emissions
ambiguous-generic = leans to the vocabulary's average (central / unmarked)
direction; redundant = near-duplicate of another seed

── Confounds ──
does the axis track a nuisance (not the construct)?

• Frequency -0.536 (confound, 28.7% var) -- leans toward more frequent words;
  partly rewards common vocabulary over the construct.
• Length 0.089 (negligible, 0.8% var) -- word length does not affect the score.
• Generic direction 0.354 (confound, 12.5% var) -- leans into the vocabulary's
  average direction; partly measures geometric centrality, not the construct.
```

How to read it:

- **Coherence** — mean cosine of each seed to the list's centroid (tightness, $0$–$1$). Low means the seeds point in many directions.
- **Word audit** — every seed is an `anchor` unless it is `redundant` (near-duplicate of another seed), `weak-offcentre` (a low outlier within its pole), `ambiguous-generic` (leans toward the vocabulary's average, unmarked direction), `ambiguous-xpole` (closer to the *other* pole, bipolar only), or `nonword` (fails the hunspell spellcheck). The table is worst-first and capped per flag; the full per-word table stays in `$cards`.
- **Confounds** — does the axis track word **frequency**, **length**, or the **generic** direction instead of the construct?

For a **bipolar** dictionary you get one card per pole plus a *contrast* (how opposed the poles are):

```r
energy <- data.frame(
  word  = c("renewable", "solar", "wind", "sustainable", "clean",
            "efficiency", "conservation", "hydroelectric",
            "fossil", "coal", "oil", "petroleum", "pollution",
            "emissions", "drilling", "combustion"),
  score = c(rep(1, 8), rep(-1, 8)))

dictionary_eval(energy, embeddings = "vdic_data/cc.en.300.vec")
```

```
── Dictionary report card: score ──────────────────────────────────
ℹ 16 seeds (8 high / 8 low) from 16/16 stems matched

── High pole (+) ──
Coherence 0.649 (mean seed-to-centroid cosine; tightness 0-1, size-sensitive)
✔ Word audit: all 8 seeds are clean anchors

── Low pole (-) ──
Coherence 0.649 (mean seed-to-centroid cosine; tightness 0-1, size-sensitive)
Word audit: 7/8 clean anchors; 1 flagged (weak-offcentre 1)
  word       flag           metric value note
  combustion weak-offcentre z      -2.10
weak-offcentre = low outlier within its pole

── Contrast ──
Pole-means cosine 0.626 (poles still share much direction; lower = more opposed)
```

### `dictionary_compare()` — are two operationalizations the same?

When you have rival seed lists for a construct, compare them. You get the **heading cosine** between every pair of axes, a threshold-free **cross-seed AUC** (can each axis rank its own seeds above a rival's?), and — for exactly two lists — the **disagreement word-lists**: what each captures that the other misses, ranked by the *capturing* list's own score (the terms most central to it, not merely the widest gap).

```r
weather   <- c("hurricane", "flood", "drought", "wildfire", "heatwave", "storm",
               "cyclone", "typhoon", "flooding", "wildfires", "tornado",
               "blizzard", "hurricanes", "storms")
technical <- c("emissions", "carbon", "dioxide", "greenhouse", "methane",
               "atmospheric", "concentrations", "anthropogenic", "warming",
               "temperatures", "fossil", "fuels", "combustion", "aerosols",
               "emission", "radiative")

dictionary_compare(list(weather = weather, technical = technical),
                   embeddings = "vdic_data/cc.en.300.vec")
```

```
── Dictionary comparison: score ───────────────────────────────────
ℹ weather vs technical: heading |cos| 0.2 -- near-orthogonal -- distinct constructs

── Heading |cos| (1 = same axis, 0 = orthogonal) ──
          weather technical
weather       1.0       0.2
technical     0.2       1.0

── Cross-seed AUC (row axis ranks ITS seeds above the column's; ~.5 = interchangeable, -- = self) ──
          weather technical
weather   -       1.000
technical 1.000   -

── weather captures, technical misses (top gaps, ranked by weather's score) ──
       word projA projB   gap
     floods 0.770 0.198 0.572
      rains 0.696 0.136 0.560
 torrential 0.683 0.054 0.629
  tornadoes 0.628 0.118 0.510
  downpours 0.613 0.086 0.527
  mudslides 0.611 0.084 0.527

── technical captures, weather misses (top gaps, ranked by technical's score) ──
       word projA projB    gap
      gases 0.112 0.831 -0.719
 pollutants 0.190 0.686 -0.496
     sulfur 0.067 0.662 -0.595
   nitrogen 0.014 0.640 -0.626
    emitted 0.125 0.612 -0.487
     gasses 0.036 0.590 -0.554
```

Here the two lists are **near-orthogonal** ($|\cos| = 0.2$): "extreme-weather events" and "atmospheric chemistry" are genuinely different constructs, and the word-lists show exactly how.

> **More than two lists?** `dictionary_compare()` scales up: instead of a pair's disagreement lists it prints each list's **one-vs-rest signature** (the words it scores high that *every* rival scores lower), plus a readability **summary** that orders the matrices so near-duplicates sit together, names the broadest and most distinct lists, the closest pair, and flags any pair with $|\cos| \ge 0.9$ as a merge candidate. (Unipolar lists only.)

### `dictionary_suggest()` — which words to add

Two add-modes: **sharpen** (the strongest on-pole words, which tighten the axis) and **broaden** (on-theme words in new directions, via greedy farthest-point). For a bipolar list both poles are handled.

```r
dictionary_suggest(energy, embeddings = "vdic_data/cc.en.300.vec")
```

```
── Words to add: score ─────────────────────────────────────────────
── Pole: high (+) ──
Sharpen the axis (strongest on-pole words)
         word axis_proj nearest_seed
 photovoltaic     0.344        solar
   excellence     0.314  sustainable
! only 2 on-theme candidates cleared the filters (asked for 20): the
  clean-energy region is thin, so this pole needs more or better seeds

── Pole: low (-) ──
Sharpen the axis (strongest on-pole words)
         word axis_proj nearest_seed
        fumes     0.357    pollution
        crude     0.345          oil
      barrels     0.342          oil
        shale     0.325         coal
 hydrocarbons     0.315    petroleum
          gas     0.311          oil

Broaden coverage (on-theme, new directions)
        word axis_proj nearest_seed
       shale     0.325         coal
    spillage     0.303    pollution
     gasoline     0.274          oil
      vapors     0.273   combustion
```

The suggestions are candidates for *you* to judge, not automatic edits — note `excellence` slipping into the thin high pole. A sparse pole returns few words **and warns** rather than padding the list with noise.

### Guidelines (the curation loop)

1. **Start focused.** One construct, ideally one facet at a time — narrow lists cohere; grab-bags don't.
2. **Run `dictionary_eval()`.** Drop `redundant` near-duplicates, reconsider `ambiguous-generic` words, inspect `weak-offcentre` outliers and any cross-pole leakage. Watch the confounds — a strong frequency or generic-direction reading means the axis is partly measuring a nuisance.
3. **Compare rivals with `dictionary_compare()`** when you're unsure two lists capture the same thing. Near-orthogonal axes are distinct constructs; $|\cos| \ge 0.9$ pairs are duplicates to merge.
4. **Extend with `dictionary_suggest()`** — *sharpen* to raise coherence, *broaden* to widen coverage, then re-evaluate.
5. **Iterate** until the card is clean. Then build.

> Curation reads the **operative axis** (raw seed geometry), never a learned/regularized one, and never edits your list — every decision stays yours. The tools default to a `hunspell` spellcheck and respect `language` (e.g. `language = "es"` keeps accented words and validates them against Spanish rather than silently filtering to English).

---

## Step 2 — Build the vec-tionary

Once the seeds are clean, learn the axis:

```r
climate_vect <- vectionary_builder(
  dictionary = climate,
  embeddings = "vdic_data/cc.en.300.vec",
  language   = "en")
# Automatically saved to ./vectionary.rds  (pass save_path = NULL to skip)

# A bipolar (graded) dictionary builds the same way:
energy_vect <- vectionary_builder(energy, "vdic_data/cc.en.300.vec",
                                  save_path = "energy_vectionary.rds")
```

`vectionary_builder()` is highly configurable — regularization method, lambda selection, vocabulary/stem expansion, spell-checking, and stopword handling. See **Builder Options** below.

---

## Step 3 — Apply the vec-tionary

A built vec-tionary scores text through metric methods on the object:

```r
# Single text (returns a named list, one element per dimension)
climate_vect$mean("Rising sea levels are displacing coastal communities")
#> $score
#> [1] 0.5413

# All metrics at once
climate_vect$metrics("Record droughts and wildfires devastated the region")
#> $mean$score      [1] 0.6072
#> $msp$score       [1] 0.4128
#> $sd$score        [1] 0.2233
#> $se$score        [1] 0.0993
#> $top_10$score    [1] 0.6072
#> $top_20$score    [1] 0.6072
```

### Batch analysis

```r
texts <- c("Carbon emissions hit a new high",
           "The local football team won",
           "Coastal flooding forced evacuations")

vectionary_analyze(climate_vect, texts, metric = "mean")
#> $score
#> [1] 0.5894 0.0712 0.5331
```

### Topic classification

Pass `alpha` to flag documents that exceed a per-dimension statistical threshold (a one-tailed t-test on the corpus of document means):

$$\text{threshold}_d = \bar{x}_d + t_{1-\alpha,\, n-1} \cdot s_d$$

```r
result <- vectionary_analyze(climate_vect, texts, metric = "mean", alpha = 0.15)
result$score         # numeric scores
result$score_topic   # logical: above threshold?
attr(result, "threshold")
as.data.frame(result)
```

Requires ≥ 2 documents. Lower `alpha` → stricter threshold.

### Diagnose

Check that your seed words rank near the top of the projections:

```r
climate_vect$diagnose()
# Or: vectionary_diagnose(climate_vect, n = 50)
```

### Save and load

```r
saveRDS(climate_vect, "climate_vectionary.rds")   # ~3 MB, no embeddings needed
climate_vect <- readRDS("climate_vectionary.rds")
```

---

## Metrics

Each vec-tionary exposes these methods via `$`:

| Method | Description |
|--------|-------------|
| `$mean(text)` | Arithmetic mean of word projections |
| `$msp(text)` | Mean square projection |
| `$sd(text)` | Standard deviation of projections (sample, $n-1$) |
| `$se(text)` | Standard error of the mean |
| `$top_10(text)` | Mean of the 10 highest projections |
| `$top_20(text)` | Mean of the 20 highest projections |
| `$metrics(text)` | All six at once |
| `$diagnose(n)` | Diagnostic report (top words, seed-word ranks) |

All accept a single string or a character vector (batch). Results are named lists (one element per dimension); use `as.data.frame()` to convert. Scores are frequency-sensitive and length-normalized, and are **not** comparable across different vec-tionaries.

## Builder Options

### Regularization methods

```r
vectionary_builder(dict, emb, method = "ridge")                       # default: dense axes
vectionary_builder(dict, emb, method = "elastic_net", l1_ratio = 0.5) # balanced sparsity
vectionary_builder(dict, emb, method = "lasso")                       # maximum sparsity
vectionary_builder(dict, emb, method = "duan")                        # unit-norm constraint
```

### Lambda (regularization strength)

```r
vectionary_builder(dict, emb, lambda = "gcv")               # default: auto-selected
vectionary_builder(dict, emb, lambda = 0.1)                 # fixed
vectionary_builder(dict, emb, lambda = c(0.01, 0.1, 0.5, 1))# grid: best validity + differentiation
```

GCV is closed-form for ridge; for lasso/elastic net the builder uses `glmnet::cv.glmnet`. A numeric vector is searched for the value that clears `min_validity` (AUC for binary, $R^2$ for continuous) with the lowest axis correlation.

### Vocabulary and stem expansion

```r
# Add the top-N highest-projecting words to the dictionary, then rebuild
vectionary_builder(dict, emb, expand_vocab = 100, expand_positive = TRUE)

# Expand stem patterns to all matching inflections in the vocabulary
vectionary_builder(c("warm*", "pollut*"), emb, expand_stem = TRUE)
# warm -> warming, warmer, ...   pollut -> pollution, pollutants, ...
```

### Language, spell-checking, stopwords, reproducibility

```r
vectionary_builder(dict, emb, language = "pt")                 # any wooorm/dictionaries language
vectionary_builder(dict, emb, spellcheck = FALSE)              # default: on
vectionary_builder(dict, emb, remove_stopwords = c("the","a")) # custom list
vectionary_builder(dict, emb, binary_word = FALSE)             # use graded scores as-is
vectionary_builder(dict, emb, seed = 574)                      # reproducible build
```

## Image analysis (multi-modal)

With SigLIP embeddings, a **text** dictionary learns an axis in a shared text–image space, then scores images directly:

```r
# Requires Python with transformers/torch via reticulate
download_embeddings(model = "siglip")

vect <- vectionary_builder(climate, embeddings = "siglip", modality = "multimodal")

analyze_image(vect, images = c("flood.jpg", "solar_farm.jpg"))  # data frame of scores
analyze_text(vect,  text   = "a wildfire at night")             # same axis, text side
```

Image embedding/analysis require Python (`transformers`, `torch`, `Pillow`, `sentencepiece`) reachable through `reticulate`; text-only workflows are pure R.

## Embedding sources

| Model | Languages (built-in) | Download |
|-------|----------------------|----------|
| FastText | `pt`, `en`, `es` | `download_embeddings("en", "fasttext")` |
| word2vec | `en` | `download_embeddings("en", "word2vec")` |
| GloVe | `en` | `download_embeddings("en", "glove")` |
| SigLIP 2 | multi-modal (text + image) | `download_embeddings(model = "siglip")` |

FastText publishes vectors for 157 languages; `download_embeddings()` fetches `pt`/`en`/`es` directly. For any other language, download the `cc.<lang>.300.vec` file and pass its path to `vectionary_builder()` (the builder's `language` argument, used for stopwords and spell-checking, accepts any [wooorm/dictionaries](https://github.com/wooorm/dictionaries) language).

## Build pipeline

`vectionary_builder()` runs a 7-step pipeline:

1. **Filter vocabulary** — drop non-alphabetic tokens (Unicode `\p{L}`), case-variant duplicates, stopwords, and misspellings (hunspell).
2. **Expand stems** — `warm*` → all inflected forms (if `expand_stem = TRUE`).
3. **Select lambda** — GCV (ridge), cross-validation (lasso/elastic net), or grid search.
4. **Learn axes** — per dimension, solve a regression from seed embeddings to scores.
5. **Expand vocabulary** — optionally add the top-N highest-projecting words.
6. **Rebuild** — re-learn on the enlarged dictionary if expansion happened.
7. **Project full vocabulary** — one matrix multiply; package and save the `Vec-tionary`.

## Technical details

- Embeddings are **unit-normalized** before learning and projection, so dot products are cosine similarities (this removes FastText/word2vec frequency-norm bias).
- All learned axes (ridge, elastic net, lasso, Duan) are **unit-normalized** after solving, so a word's projection is a pure cosine in $[-1, 1]$.
- The Duan et al. (2025) method uses constrained optimization (`alabama::auglag`) with a unit-norm axis constraint and no regularization.
- GCV (Golub et al., 1979) minimizes $GCV(\lambda) = \frac{n^{-1}\, \lVert y - S_\lambda y \rVert^2}{(1 - \mathrm{tr}(S_\lambda)/n)^2}$ via SVD of the embedding matrix.
- Standard deviations / errors use **sample** ($n-1$) statistics; topic thresholds use a one-tailed t-test.
- RNG state is saved and restored around every `set.seed()`, so building never perturbs your random stream.

## Requirements

- R ≥ 3.5.0
- R.utils, cli, MASS, glmnet, data.table (installed automatically)
- Optional: `hunspell` (spell-checking, used by the dictionary tools and builder), `alabama` (Duan method), `reticulate` + Python (image analysis)

## Citation

```bibtex
@software{vdic,
  title  = {vdic: Build and Use Vec-tionaries for Text and Image Analysis},
  author = {Leonardo Dantas},
  year   = {2026},
  note   = {R package version 1.2.0},
  url    = {https://github.com/leofdantas/vdic}
}
```

## Related work

Generalizes the vec-tionary approach introduced in:

- Duan, Z., Shao, A., Hu, Y., Lee, H., Liao, X., Suh, Y. J., Kim, J., et al. (2025). "Constructing Vec-Tionaries to Extract Message Features from Texts: A Case Study of Moral Content." *Political Analysis*, 1–21. doi:10.1017/pan.2025.6.
- https://github.com/ZeningDuan/vMFD

## License

MIT

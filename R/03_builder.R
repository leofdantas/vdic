# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  03_builder.R — Vec-tionary Construction Engine                            ║
# ║                                                                            ║
# ║  This file implements the core pipeline for building vec-tionaries:        ║
# ║    1. Filter embedding vocabulary (spellcheck, stopwords, non-alphabetic)  ║
# ║    2. Expand stem patterns (e.g., "abandon*" → all inflections)            ║
# ║    3. Select regularization parameter lambda (GCV, CV, or grid search)     ║
# ║    4. Learn semantic axes via regression (ridge / elastic net / Duan)      ║
# ║    5. Optionally expand dictionary with high-projection words              ║
# ║    6. Project full embedding vocabulary onto learned axes                  ║
# ║    7. Package into S3 "Vec-tionary" object and save                        ║
# ║                                                                            ║
# ║  Mathematical core:                                                        ║
# ║    Ridge:       $a = (W^T W + \lambda I)^{-1} W^T y$                      ║
# ║    Elastic net: glmnet with L1+L2 penalty                                 ║
# ║    LASSO:       glmnet with L1 penalty (l1_ratio = 1)                     ║
# ║    Duan (2025): $\min ||Wm - y||^2$ s.t. $||m|| = 1$                      ║
# ║                                                                            ║
# ║  All word embeddings are unit-normalized before axis learning so that      ║
# ║  projections are cosine-similarity-based rather than magnitude-biased.     ║
# ║  Axes are NOT unit-normalized — their scale encodes dictionary scores.     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

#' Build a vec-tionary from dictionary and embeddings
#'
#' @description
#' Builds a vector-based dictionary (vec-tionary) by learning axes in embedding
#' space from seed words in the input dictionary, then projects ALL words from
#' the embeddings onto these axes. This replicates the Duan et al. (2025) approach
#' where the vectionary can score any word in the full embedding vocabulary.
#' The resulting file size depends on the embedding vocabulary (typically ~3 MB
#' as a compressed RDS).
#'
#' @param dictionary Either a data frame or a character vector of words.
#'   - **Data frame**: Must have a 'word' column plus one or more dimension columns
#'     (e.g., 'care', 'fairness', 'sentiment'). ALL words are used to learn axes.
#'   - **Character vector**: A simple list of seed words. Creates a binary dictionary
#'     where all words score 1 on each dimension specified (or "score" if dimensions=NULL).
#' @param embeddings Path to word embeddings file. Can be FastText .vec format,
#'   word2vec .bin format, or GloVe .txt format.
#' @param language Language code for stopwords and spell checking (default: "en").
#'   Any language supported by the wooorm/dictionaries repository can be used
#'   (e.g., "en", "pt", "es", "fr", "de", "it", "nl", "ru"). When spellcheck = TRUE,
#'   the hunspell dictionary for this language is automatically downloaded and cached
#'   in vdic_data/dictionaries/ (in the working directory) if not already available. Stopword lists are built-in
#'   for "en", "pt", and "es"; other languages skip stopword removal unless a custom
#'   list is provided via \code{remove_stopwords}.
#'   See https://github.com/wooorm/dictionaries for all available languages.
#' @param dimensions Character vector of dimension names to build axes for.
#'   If NULL (default), uses all columns in dictionary except 'word'.
#' @param binary_word Logical. If TRUE (default), treats dictionary as binary:
#'   converts all non-zero values to 1 and blanks/NAs/NaNs/nulls to 0.
#'   If FALSE, uses continuous scores as provided in dictionary.
#' @param method Regularization method for axis learning (default: "ridge"). Options:
#'   \itemize{
#'     \item "ridge": Ridge regression (L2 penalty) - smooth, dense axes
#'     \item "elastic_net": Elastic net (L1 + L2 penalty) - balanced sparsity
#'     \item "lasso": LASSO (L1 penalty) - maximum sparsity, sets l1_ratio=1
#'     \item "duan": Duan et al. (2025) method - constrained nonlinear optimization
#'       with unit norm constraint, no regularization. Replicates the vMFD approach.
#'   }
#' @param l1_ratio Elastic net mixing parameter (default: 0.5). Only used when
#'   method="elastic_net". Value between 0 (pure ridge) and 1 (pure lasso).
#'   Controls sparsity of learned axes. Higher values produce sparser axes with
#'   more zero coefficients. Ignored for method="ridge" or "lasso".
#' @param lambda Regularization parameter. Can be:
#'   \itemize{
#'     \item "gcv" (default): Use Generalized Cross-Validation (Golub et al., 1979) to select
#'       optimal lambda. Only works with method="ridge".
#'     \item Numeric value (e.g., 0.5): Use this specific lambda
#'     \item Numeric vector (e.g., c(0.01, 0.1, 1)): Test these values and select optimal
#'   }
#'   Higher lambda creates more regularized axes. When multiple values are provided,
#'   the optimal lambda is selected based on validity (R² or AUC) and axis differentiation.
#' @param min_validity Minimum validity threshold (default: 0.75 = 75%).
#'   Only used when lambda is a numeric vector (ignored for "gcv").
#'   For continuous dictionaries: R² between scores and projections.
#'   For binary dictionaries: AUC (0.5 = random, 1.0 = perfect separation).
#'   Lambda candidates below this threshold are rejected unless none pass.
#' @param expand_vocab Integer or NULL (default). If set, expands the training
#'   dictionary before learning axes. First learns preliminary axes from seed words,
#'   then finds the top-N words with highest projections onto those axes. These
#'   words are added to the dictionary (with their projection as the score), and
#'   final axes are learned from the expanded dictionary. This improves axis quality
#'   by increasing the training signal.
#' @param spellcheck Logical (default: TRUE). If TRUE, filters the embedding
#'   vocabulary using hunspell spell checking (with the dictionary matching
#'   the \code{language} parameter) before projection. This removes non-words,
#'   typos, symbols, and web artifacts commonly found in embeddings trained on
#'   Common Crawl data. The spell check is applied first, so expand_stem and
#'   expand_vocab will only find valid dictionary words. Requires the hunspell
#'   and data.table packages.
#' @param expand_positive Logical (default: TRUE). If TRUE and expand_vocab is set,
#'   only adds words with positive projections onto the preliminary axes. Recommended
#'   for binary dictionaries where negative projections are not semantically meaningful.
#' @param expand_stem Logical (default: FALSE). If TRUE, expands dictionary words
#'   ending with * (e.g., "abandon*") to all matching words found in the embeddings
#'   (e.g., "abandon", "abandoned", "abandoning", "abandonment"). The expanded words
#'   inherit the same scores as the stem pattern. This is useful for dictionaries
#'   built with word stems or regex-style patterns.
#' @param remove_stopwords Logical (default: TRUE). If TRUE, removes stopwords
#'   (based on the \code{language} parameter) from the embedding space so that
#'   the vectionary is not projected to these words. Useful for simplicity and
#'   interpretability, as stopwords are the most common words and their scores
#'   can bias the output of vectionary_analyze(). Can also be a character vector
#'   of custom stopwords.
#' @param save_path Where to save the vectionary. Can be:
#'   \itemize{
#'     \item A filename ending in `.rds` (e.g., "my_vect.rds") - saves to working directory
#'     \item A full path with filename (e.g., "~/data/my_vect.rds") - saves to that path
#'     \item A directory path (e.g., getwd()) - saves as "vectionary.rds" in that directory
#'     \item NULL - does not save
#'   }
#'   Default: current working directory (saves as "vectionary.rds").
#' @param verbose Logical (default: TRUE). If TRUE, prints step-by-step progress
#'   messages: dictionary summary, embedding loading, lambda selection, axis
#'   learning, vocabulary expansion, projection, and save confirmation.
#' @param seed Integer seed for reproducibility. Controls random operations
#'   (Duan method initialization, AUC validation sampling). If NULL (default),
#'   a random seed is generated and stored in \code{metadata$seed} so the build
#'   can be reproduced later.
#'
#' @return A vectionary object containing:
#'   \itemize{
#'     \item axes: Named list of dimension vectors in embedding space
#'     \item word_projections: Pre-computed projections for ALL words in the embeddings
#'     \item dimensions: Names of dimensions
#'     \item metadata: Build information (method, embedding source, vocab_size, etc.)
#'   }
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Portuguese moral foundations example
#' dictionary <- data.frame(
#'   word = c("proteger", "cuidar", "ajudar", "machucar", "prejudicar"),
#'   care = c(0.9, 0.8, 0.7, -0.8, -0.7),
#'   fairness = c(0.1, 0.2, 0.3, 0.0, 0.1)
#' )
#'
#' # Default: GCV (Generalized Cross-Validation) for optimal lambda
#' my_vect <- vectionary_builder(
#'   dictionary = dictionary,
#'   embeddings = "vdic_data/cc.pt.300.vec",
#'   dimensions = c("care", "fairness")
#' )
#'
#' # Specify lambda manually (single value)
#' my_vect <- vectionary_builder(
#'   dictionary = dictionary,
#'   embeddings = "vdic_data/cc.pt.300.vec",
#'   dimensions = c("care", "fairness"),
#'   lambda = 0.5
#' )
#'
#' # Select from custom range
#' my_vect <- vectionary_builder(
#'   dictionary = dictionary,
#'   embeddings = "vdic_data/cc.pt.300.vec",
#'   dimensions = c("care", "fairness"),
#'   lambda = c(0.01, 0.1, 0.5, 1)
#' )
#'
#' # Use elastic net for sparse axes
#' my_vect_sparse <- vectionary_builder(
#'   dictionary = dictionary,
#'   embeddings = "vdic_data/cc.pt.300.vec",
#'   dimensions = c("care", "fairness"),
#'   method = "elastic_net",
#'   l1_ratio = 0.8,  # More LASSO-like (sparser)
#'   lambda = c(0.1, 0.5, 1)
#' )
#'
#' # Use the vec-tionary
#' my_vect$mean("Devemos proteger as pessoas vulneráveis")
#'
#' # Vectionary is automatically saved to save_path (default: working directory)
#' # To skip saving, use save_path = NULL
#'
#' # Simple word list (character vector) - creates binary dictionary
#' care_words <- c("protect", "care", "help", "safe", "harm", "hurt")
#' care_vect <- vectionary_builder(
#'   dictionary = care_words,
#'   embeddings = "vdic_data/cc.en.300.vec",
#'   dimensions = "care"
#' )
#' }

# ============================================
# Main function ====
# ============================================

vectionary_builder <- function(
  dictionary,
  embeddings,
  language = "en",
  dimensions = NULL,
  binary_word = TRUE,
  method = "ridge",
  l1_ratio = 0.5,
  lambda = "gcv",
  min_validity = 0.75,
  expand_vocab = NULL,
  spellcheck = TRUE,
  expand_positive = TRUE,
  expand_stem = FALSE,
  remove_stopwords = TRUE,
  save_path = getwd(),
  verbose = TRUE,
  seed = NULL
) {

#- Input validation and conversion ----

  # The dictionary can arrive in two forms:
  #   1. Character vector of words → converted to a binary data frame (all words score 1)
  #   2. Data frame with 'word' column + numeric dimension columns
  # Both are validated and normalized before entering the pipeline.

  ##- Handle character vector input ----
  # Convert character vector to binary data frame where every word scores 1
  # on each requested dimension. This is a convenience shorthand for users
  # who just have a word list without graded scores.
  if (is.character(dictionary)) {
    if (length(dictionary) == 0) {
      stop("dictionary cannot be empty")
    }

    # Determine dimension names
    if (is.null(dimensions)) {
      dim_names <- "score"
    } else {
      dim_names <- dimensions
    }

    # Create data frame with all words scoring 1 on each dimension
    dictionary <- data.frame(word = dictionary, stringsAsFactors = FALSE)
    for (dim in dim_names) {
      dictionary[[dim]] <- 1
    }

    if (verbose) {
      cli_alert_info("Converted character vector to binary dictionary with {length(dim_names)} dimension{?s}: {paste(dim_names, collapse = ', ')}")
    }
  }

  if (!is.data.frame(dictionary)) {
    stop("dictionary must be a data frame or character vector")
  }

  if (!"word" %in% names(dictionary)) {
    stop("dictionary must have a 'word' column")
  }
  ##- Validate dictionary words ----
  # Dictionary words must be single tokens (no spaces), free of punctuation
  # (unless expand_stem=TRUE allows trailing *), and will be lowercased.
  # These constraints ensure clean matching against embedding vocabulary.
  words <- dictionary$word

  ##- Check for multi-word entries (spaces) ----
  # N-grams (e.g., "human rights") are not supported because word embeddings
  # are per-token. Future versions may support phrase embeddings.
  has_spaces <- grepl("\\s", words)
  if (any(has_spaces)) {
    bad_words <- words[has_spaces][1:min(3, sum(has_spaces))]
    stop("Dictionary contains multi-word entries (not yet supported):\n",
         "  ", paste(bad_words, collapse = ", "),
         if (sum(has_spaces) > 3) paste0(", ... (", sum(has_spaces), " total)") else "",
         "\n\nPlease use single-word entries only. ",
         "N-gram support is planned for a future version.")
  }

  ##- Detect stem patterns ----
  # Stem patterns like "abandon*" are used with expand_stem=TRUE to match all
  # inflected forms in the embeddings (e.g., "abandoned", "abandoning", etc.).
  # Trailing * is the only punctuation allowed when expand_stem=TRUE.
  is_stem <- grepl("\\*$", words)

  if (expand_stem) {
    # When expand_stem = TRUE, allow trailing * but reject other punctuation.
    # Strip the trailing * before checking so it doesn't trigger a false positive.
    words_for_check <- sub("\\*$", "", words)
    has_punct <- grepl("[[:punct:]]", words_for_check)

    if (any(has_punct)) {
      bad_words <- words[has_punct][1:min(3, sum(has_punct))]
      stop("Dictionary contains words with punctuation (other than trailing *):\n",
           "  ", paste(bad_words, collapse = ", "),
           if (sum(has_punct) > 3) paste0(", ... (", sum(has_punct), " total)") else "",
           "\n\nPlease remove punctuation from dictionary words.")
    }
  } else {
    # When expand_stem = FALSE, reject all punctuation
    has_punct <- grepl("[[:punct:]]", words)

    if (any(has_punct)) {
      bad_words <- words[has_punct][1:min(3, sum(has_punct))]

      # Check if any look like stem patterns (end with *)
      has_stems <- any(is_stem)

      stop("Dictionary contains words with punctuation:\n",
           "  ", paste(bad_words, collapse = ", "),
           if (sum(has_punct) > 3) paste0(", ... (", sum(has_punct), " total)") else "",
           "\n\nPlease remove punctuation from dictionary words.",
           if (has_stems) "\n\nIf working with word stems with regex compatibility, set expand_stem = TRUE to fill stems by all matching full words." else "")
    }
  }

  ##- Detect uppercase ----
  # Embeddings are case-insensitive in this package (all lowercased on load).
  # Warn and auto-convert rather than erroring, since mixed-case dictionaries
  # are a common user mistake, not a data quality issue.
  has_upper <- grepl("[A-Z]", words)
  if (any(has_upper)) {
    if (verbose) {
      cli_alert_warning("Dictionary contains uppercase letters. Converting to lowercase.")
    }
    dictionary$word <- tolower(dictionary$word)
  }

  if (!file.exists(embeddings)) {
    stop("Embeddings file not found: ", embeddings, "\n",
         "Download embeddings with: download_embeddings()")
  }

  ##- Resolve language ----
  if (!is.character(language) || length(language) != 1 || nchar(language) == 0) {
    stop("language must be a single non-empty character string (e.g., 'en', 'pt', 'fr')")
  }

  ##- Resolve stopwords ----
  # Stopwords can come from three sources:
  #   remove_stopwords = TRUE  → built-in list for this language (from 01_package.R)
  #   remove_stopwords = character vector → user-supplied custom list
  #   remove_stopwords = FALSE → no stopword filtering (stopwords stays NULL)
  stopwords <- NULL
  if (isTRUE(remove_stopwords)) {
    stopwords <- .get_stopwords(language)
  } else if (is.character(remove_stopwords)) {
    stopwords <- remove_stopwords
  }

  ##- Validate and process lambda parameter ----
  # Lambda controls regularization strength. Three modes:
  #   "gcv"           → automatic selection via Generalized Cross-Validation
  #   single numeric  → use that exact value (lambda_range stays NULL)
  #   numeric vector  → grid search over those values (stored in lambda_range)
  # The flags use_gcv and lambda_range determine which path Step 3 takes.
  lambda_range <- NULL
  use_gcv <- FALSE

  if (is.character(lambda)) {
    if (lambda == "gcv") {
      use_gcv <- TRUE
    } else {
      stop("lambda must be numeric, a numeric vector, or 'gcv'")
    }
  } else if (is.numeric(lambda)) {
    if (any(lambda < 0)) {
      stop("lambda values must be non-negative")
    }
    if (length(lambda) == 1) {
      lambda_range <- NULL
    } else {
      lambda_range <- lambda
    }
  } else {
    stop("lambda must be numeric, a numeric vector, or 'gcv'")
  }

  ##- Validate method parameter ----
  if (!method %in% c("ridge", "lasso", "elastic_net", "duan")) {
    stop("method must be one of: 'ridge', 'elastic_net', 'lasso', 'duan'")
  }

  # Duan method uses constrained optimization with no regularization,
  # so lambda is irrelevant. Clear all lambda-related settings.
  if (method == "duan") {
    lambda_range <- NULL
    use_gcv <- FALSE
    lambda <- NULL
  }

  if (!is.numeric(l1_ratio) || l1_ratio < 0 || l1_ratio > 1) {
    stop("l1_ratio must be a number between 0 and 1")
  }

  ##- Determine dimensions to build ----
  if (is.null(dimensions)) {
    dimensions <- setdiff(names(dictionary), "word")
  }
  if (length(dimensions) == 0) {
    stop("No dimensions specified. dictionary must have columns besides 'word'")
  }

  missing_dims <- setdiff(dimensions, names(dictionary))
  if (length(missing_dims) > 0) {
    stop("Dimensions not found in dictionary: ", paste(missing_dims, collapse = ", "))
  }

  ##- Convert to binary if requested ----
  # Binary mode (default) collapses scores to 1 (present) or 0 (absent).
  # Absent: NA, NaN, numeric 0, empty/whitespace-only strings, empty factors.
  # Present: any nonzero number, any non-empty string or factor level.
  if (binary_word) {
    dictionary <- as.data.frame(dictionary)
    for (dim in dimensions) {
      col <- dictionary[[dim]]
      if (is.character(col) || is.factor(col)) {
        col <- as.character(col)
        is_absent <- is.na(col) | trimws(col) == ""
      } else {
        is_absent <- is.na(col) | col == 0
      }
      dictionary[[dim]] <- as.integer(!is_absent)
    }
  }

  ##- Check for indistinguishable dimensions ----
  # After binary conversion (or character vector expansion), verify that
  # dimension score patterns actually differ. If all dimensions have the
  # same scores for every word, the solver receives identical input and
  # produces identical axes — a silent but serious problem.
  if (length(dimensions) > 1) {
    ref <- dictionary[[dimensions[1]]]
    all_same <- all(vapply(dimensions[-1], function(d) {
      identical(ref, dictionary[[d]])
    }, logical(1)))

    if (all_same) {
      stop(
        "All dimensions have identical score patterns — axes would be indistinguishable.\n",
        if (binary_word) {
          paste0(
            "binary_word = TRUE converted all nonzero values to 1, collapsing dimensions.\n",
            "Fix: use binary_word = FALSE for continuous-valued dictionaries, or\n",
            "     use 0 or NA for words that do NOT belong to a given dimension."
          )
        } else {
          "Ensure each dimension has a distinct pattern of scores across words."
        }
      )
    }
  }

  ##- Validate expand_vocab ----
  # Accept TRUE as shorthand for a default value (100)
  if (isTRUE(expand_vocab)) {
    expand_vocab <- 100L
  } else if (identical(expand_vocab, FALSE)) {
    expand_vocab <- NULL
  }

  ##- Generate seed if not provided ----
  # A single seed controls all random operations in the pipeline (Duan method
  # initialization, AUC validation sampling). Per-dimension seeds are derived
  # from this base seed + dimension name hash inside each function.
  # The seed is stored in metadata for exact reproducibility.
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }

  # ════════════════════════════════════════════════
  # PIPELINE
  # ════════════════════════════════════════════════

  #- Banner ----
  if (verbose) {
    cli_h1("Building Vec-tionary")
    cli_ul(c(
      "Dictionary: {nrow(dictionary)} words, {length(dimensions)} dimension{?s} ({paste(dimensions, collapse = ', ')})",
      "Embeddings: {basename(embeddings)}",
      "Language: {language}",
      "Method: {method}{if (method == 'duan') ' (Duan et al., 2025)' else ''}",
      "Spellcheck: {if (spellcheck) 'yes' else 'no'} | Stopwords: {if (!is.null(stopwords)) paste0('yes (', length(stopwords), ')') else 'no'}",
      "Expand stems: {if (expand_stem) 'yes' else 'no'} | Expand vocab: {if (!is.null(expand_vocab)) expand_vocab else 'no'}"
    ))
    cli_text("")
  }

  # Track total build time
  t_total_start <- Sys.time()

  #- Step 1: Filter embedding vocabulary ----
  # Create a cleaned temporary copy of the embeddings file. This step runs first
  # so that all downstream functions (stem expansion, axis learning, projection)
  # operate on the filtered vocabulary only. Filtering includes:
  #   - Non-alphabetic token removal (always): numbers, codes, symbols, URLs
  #   - Stopword removal (if remove_stopwords != FALSE)
  #   - Hunspell spell checking (if spellcheck = TRUE, requires data.table)
  # The filtered file is written to a temp path and cleaned up on exit.
  if (verbose) cli_h2("Filtering embedding vocabulary")
  t_step <- Sys.time()

  filtered_embeddings <- .filter_embedding_vocab(
    embeddings_path = embeddings,
    stopwords = stopwords,
    spellcheck = spellcheck,
    language = language,
    verbose = verbose
  )
  on.exit(unlink(filtered_embeddings), add = TRUE)

  if (verbose) {
    t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
    cli_alert_success("Done ({t_elapsed}s)")
  }

  #- Step 2: Expand stem patterns ----
  if (expand_stem) {
    stems_in_dict <- grepl("\\*$", dictionary$word)

    if (any(stems_in_dict)) {
      if (verbose) {
        cli_h2("Expanding stem patterns")
        cli_alert_info("Found {sum(stems_in_dict)} stem pattern{?s} in dictionary")
      }
      t_step <- Sys.time()

      dictionary <- .expand_stems(
        dictionary = dictionary,
        embeddings_path = filtered_embeddings,
        verbose = verbose
      )

      if (verbose) {
        t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
        cli_alert_success("Expanded dictionary: {nrow(dictionary)} words ({t_elapsed}s)")
      }
    } else if (verbose) {
      cli_alert_info("expand_stem = TRUE but no stem patterns (words ending with *) found in dictionary")
    }
  }

  #- Step 3: Select lambda ----
  # Three lambda selection paths (mutually exclusive):
  #   use_gcv = TRUE + ridge    → GCV closed-form (Golub et al., 1979)
  #   use_gcv = TRUE + non-ridge → glmnet cross-validation
  #   lambda_range != NULL       → grid search with validity-based selection
  # After this step, `lambda` is always a single numeric value.
  if (use_gcv) {
    if (method == "ridge") {
      if (verbose) cli_h2("Selecting lambda via GCV")
      t_step <- Sys.time()

      lambda <- .gcv_select_lambda_multi(
        dictionary = dictionary,
        embeddings_path = filtered_embeddings,
        dimensions = dimensions,
        aggregate = "median",
        verbose = verbose
      )

      if (verbose) {
        t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
        cli_alert_success("Done ({t_elapsed}s)")
      }
    } else {
      # Elastic net / lasso: use glmnet::cv.glmnet for cross-validated lambda
      if (verbose) cli_h2("Selecting lambda via cross-validation")
      t_step <- Sys.time()

      lambda <- .cv_select_lambda_glmnet(
        dictionary = dictionary,
        embeddings_path = filtered_embeddings,
        dimensions = dimensions,
        method = method,
        l1_ratio = l1_ratio,
        verbose = verbose
      )

      if (verbose) {
        t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
        cli_alert_success("Done ({t_elapsed}s)")
      }
    }
  }

  if (!is.null(lambda_range)) {
    if (verbose && !use_gcv) cli_h2("Selecting optimal lambda")
    t_step <- Sys.time()

    lambda <- .select_optimal_lambda(
      dictionary = dictionary,
      embeddings_path = filtered_embeddings,
      dimensions = dimensions,
      method = method,
      l1_ratio = l1_ratio,
      lambda_range = lambda_range,
      min_validity = min_validity,
      verbose = verbose,
      seed = seed
    )

    if (verbose) {
      t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
      cli_alert_success("Done ({t_elapsed}s)")
    }
  }

  #- Step 4: Build vec-tionary ----
  # Core step: learn one axis per dimension from dictionary word embeddings,
  # then project the full embedding vocabulary onto those axes.
  # Returns axes (regression coefficients) and word_projections (data frame).
  if (verbose) {
    cli_h2("Building vec-tionary")
    if (method == "duan") {
      cli_alert_info("Method: Duan et al. (2025) - constrained optimization")
    } else {
      cli_alert_info("Method: {method} | Lambda: {lambda}")
    }
  }
  t_step <- Sys.time()

  result <- .build_vectionary_internal(
    dictionary = dictionary,
    embeddings_path = filtered_embeddings,
    dimensions = dimensions,
    method = method,
    lambda = lambda,
    l1_ratio = l1_ratio,
    project_full_vocab = TRUE,
    verbose = verbose,
    seed = seed
  )

  if (verbose) {
    t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
    cli_alert_success("Done ({t_elapsed}s)")
  }

  #- Step 5: Expand vocabulary ----
  # Optional: find the top-N words from the full vocabulary that project most
  # strongly onto the preliminary axes and add them to the dictionary. This
  # bootstraps the training signal beyond the original seed words.
  # If expansion occurs, Step 6 rebuilds the axes with the enlarged dictionary.
  expanded_count <- 0
  original_dict_size <- nrow(dictionary)

  if (!is.null(expand_vocab) && expand_vocab > 0) {
    if (verbose) {
      cli_h2("Expanding vocabulary")
      cli_alert_info("Finding top {expand_vocab} words with highest projections")
    }
    t_step <- Sys.time()

    expanded_df <- .expand_vocabulary(
      word_projections = result$word_projections,
      dictionary = dictionary,
      dimensions = dimensions,
      n_expand = expand_vocab,
      positive_only = expand_positive,
      verbose = verbose
    )

    if (verbose) {
      t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
      cli_alert_success("Done ({t_elapsed}s)")
    }

    #- Step 6: Rebuild with expanded dictionary ----
    if (nrow(expanded_df) > 0) {
      expanded_count <- nrow(expanded_df)
      dictionary <- rbind(dictionary, expanded_df)

      if (verbose) {
        cli_h2("Rebuilding with expanded dictionary")
        cli_alert_info("Added {expanded_count} words, new dictionary size: {nrow(dictionary)}")
      }
      t_step <- Sys.time()

      result <- .build_vectionary_internal(
        dictionary = dictionary,
        embeddings_path = filtered_embeddings,
        dimensions = dimensions,
        method = method,
        lambda = lambda,
        l1_ratio = l1_ratio,
        project_full_vocab = TRUE,
        verbose = verbose,
        seed = seed
      )

      if (verbose) {
        t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
        cli_alert_success("Done ({t_elapsed}s)")
      }
    }
  }

  word_projections <- result$word_projections

  #- Convert to Vec-tionary S3 object ----
  # Package the learned axes and projections into an S3 object of class
  # "Vec-tionary". Metadata records every build parameter for reproducibility.
  # The $ operator (defined in 04_vectionary.R) provides method dispatch
  # (e.g., vect$mean("text")) and field access (e.g., vect$dimensions).
  vect <- structure(
    list(
      axes = result$axes,
      word_projections = word_projections,
      dimensions = dimensions,
      metadata = list(
        method = method,
        l1_ratio = if (method == "elastic_net") l1_ratio else if (method == "lasso") 1.0 else NULL,
        binary_word = binary_word,
        lambda = lambda,
        expand_vocab = expand_vocab,
        vocab_size = nrow(word_projections),
        embeddings_source = basename(embeddings),
        language = language,
        stopwords_removed = !is.null(stopwords),
        stopwords_language = if (!is.null(stopwords)) language else NULL,
        stopwords_count = if (!is.null(stopwords)) length(stopwords) else 0L,
        spellcheck = spellcheck,
        seed_words = tolower(unique(dictionary$word)),
        seed_words_per_dim = lapply(setNames(dimensions, dimensions), function(d) {
          scores <- dictionary[[d]]
          tolower(dictionary$word[!is.na(scores) & scores != 0])
        }),
        seed_words_count = original_dict_size,
        expanded_words = expanded_count,
        training_words_count = nrow(dictionary),
        words_found = result$words_found,
        seed = seed,
        build_date = Sys.time(),
        package_version = as.character(utils::packageVersion("vdic"))
      )
    ),
    class = "Vec-tionary"
  )

  if (verbose) {
    pct <- round(100 * result$words_found / nrow(dictionary), 1)
    t_total <- round(as.numeric(difftime(Sys.time(), t_total_start, units = "secs")), 1)

    cli_h2("Summary")
    cli_alert_success("Vec-tionary built successfully! ({t_total}s)")
    if (expanded_count > 0) {
      cli_alert_info("Training: {original_dict_size} seed words + {expanded_count} expanded = {nrow(dictionary)} total")
    }
    cli_alert_info("Vocabulary: {nrow(word_projections)} words projected")
    cli_alert_info("Dictionary coverage: {result$words_found}/{nrow(dictionary)} words found ({pct}%)")
  }

  #- Save vectionary ----
  # Save with xz compression (best ratio for numeric data). The save_path
  # can be a filename ("my_vect.rds"), full path, or directory.
  if (!is.null(save_path)) {
    if (verbose) {
      cli_h2("Saving vec-tionary")
    }

    # Determine save file path
    if (grepl("\\.rds$", save_path, ignore.case = TRUE)) {
      # User provided a filename (ends with .rds)
      # If it's just a filename, prepend working directory
      if (dirname(save_path) == ".") {
        save_file <- file.path(getwd(), save_path)
      } else {
        save_file <- save_path
      }
    } else {
      # User provided a directory, append default filename
      save_file <- file.path(save_path, "vectionary.rds")
    }

    saveRDS(vect, file = save_file, compress = "xz")

    file_size_mb <- file.info(save_file)$size / 1024^2
    if (verbose) {
      cli_alert_success("Saved to: {.path {save_file}} ({round(file_size_mb, 1)} MB)")
    }
  }

  return(vect)
}

#' Select optimal lambda using Generalized Cross-Validation
#'
#' @description
#' Uses GCV (Golub et al., 1979) to select the optimal regularization parameter
#' for ridge regression. GCV approximates leave-one-out cross-validation using
#' a closed-form solution based on the smoother matrix trace.
#'
#' For ridge regression with smoother matrix
#' \eqn{S = W(W'W + \lambda I)^{-1} W'}{S = W(W'W + lambda*I)^-1 W'},
#' the GCV criterion is:
#'
#' \deqn{GCV(\lambda) = \frac{n^{-1} ||y - S y||^2}{(1 - tr(S)/n)^2}}{GCV(lambda) = (RSS/n) / (1 - tr(S)/n)^2}
#'
#' The effective degrees of freedom is
#' \eqn{tr(S) = \sum_j d_j^2 / (d_j^2 + \lambda)}{tr(S) = sum(d_j^2 / (d_j^2 + lambda))}
#' where \eqn{d_j} are singular values of \eqn{W}.
#'
#' @param dictionary Data frame with 'word' column and dimension score columns
#' @param embeddings_path Path to word embeddings file
#' @param dimensions Character vector of dimension names to analyze. If NULL,
#'   uses all numeric columns except 'word'.
#' @param lambda_seq Numeric vector of lambda values to evaluate. If NULL, uses
#'   a logarithmic sequence from 1e-4 to 1e4.
#' @param aggregate How to combine optimal lambdas across dimensions:
#'   "median" (default), "mean", "min", "max", or "none" (return all)
#' @param verbose Print progress messages
#'
#' @return If aggregate != "none": single optimal lambda value
#'         If aggregate = "none": named list with optimal lambda per dimension
#'         and diagnostic data
#'
#' @references
#' Golub, G. H., Heath, M., & Wahba, G. (1979). Generalized cross-validation as
#' a method for choosing a good ridge parameter. Technometrics, 21(2), 215-223.
#'
#' Fan, J. (2026). ORF 525: Statistical Foundations of Data Science. Princeton University.
#'
#' @keywords internal
.gcv_select_lambda_multi <- function(dictionary, embeddings_path,
                              dimensions = NULL,
                              lambda_seq = NULL,
                              aggregate = c("median", "mean", "min", "max", "none"),
                              verbose = TRUE) {

  aggregate <- match.arg(aggregate)

  # ── Input validation ──
  if (!file.exists(embeddings_path)) {
    stop("Embeddings file not found: ", embeddings_path)
  }

  if (!"word" %in% names(dictionary)) {
    stop("Dictionary must have a 'word' column")
  }

  # Determine dimensions
  if (is.null(dimensions)) {
    dimensions <- setdiff(names(dictionary), "word")
    dimensions <- dimensions[sapply(dictionary[dimensions], is.numeric)]
  }

  if (length(dimensions) == 0) {
    stop("No numeric dimension columns found in dictionary")
  }

  if (verbose) {
    cli::cli_alert_info("Analyzing {length(dimensions)} dimension{?s}")
  }

  # Run GCV independently for each dimension. Each dimension has its own
  # optimal lambda because different semantic axes may need different amounts
  # of regularization depending on how many seed words define them.
  results <- list()

  for (dim in dimensions) {
    if (verbose) {
      cli::cli_progress_step("Processing dimension: {dim}", spinner = TRUE)
    }

    result <- tryCatch(
      .gcv_select_lambda_for_dimension(
        dictionary = dictionary,
        dimension = dim,
        embeddings_path = embeddings_path,
        lambda_seq = lambda_seq,
        verbose = FALSE
      ),
      error = function(e) {
        if (verbose) {
          cli::cli_alert_warning("Failed for dimension '{dim}': {e$message}")
        }
        return(NULL)
      }
    )

    if (!is.null(result)) {
      results[[dim]] <- result
    }

    if (verbose) {
      cli::cli_progress_done()
    }
  }

  if (length(results) == 0) {
    stop("GCV failed for all dimensions")
  }

  # Extract per-dimension optimal lambdas into a named vector for aggregation
  optimal_lambdas <- sapply(results, function(r) r$optimal_lambda)

  if (verbose) {
    cli::cli_text("")
    cli::cli_alert_success("GCV complete")
    for (dim in names(optimal_lambdas)) {
      df_val <- results[[dim]]$df_values[results[[dim]]$best_idx]
      cli::cli_alert_info(
        "  {dim}: lambda = {signif(optimal_lambdas[dim], 3)} (effective df = {round(df_val, 1)})"
      )
    }
  }

  # Aggregate per-dimension lambdas into a single value. Median (default) is
  # robust to outlier dimensions. "none" returns the full diagnostic list.
  if (aggregate == "none") {
    return(list(
      optimal_lambdas = optimal_lambdas,
      aggregate_lambda = NA,
      per_dimension = results
    ))
  }

  agg_lambda <- switch(aggregate,
    "median" = median(optimal_lambdas),
    "mean" = mean(optimal_lambdas),
    "min" = min(optimal_lambdas),
    "max" = max(optimal_lambdas)
  )

  if (verbose) {
    cli::cli_text("")
    cli::cli_alert_success(
      "Recommended lambda ({aggregate}): {signif(agg_lambda, 3)}"
    )
  }

  return(agg_lambda)
}

# ============================================
# Internal implementation functions ====
# ============================================

# All functions below are internal (not exported). They are called by
# vectionary_builder() and .gcv_select_lambda_multi() to implement the pipeline steps.

#-- Shared helpers ----
# Small utility functions used by multiple internal functions to avoid
# duplicating logic (header parsing, vector normalization).

#' Parse the first line of an embeddings file to detect a header
#'
#' @description
#' FastText .vec files start with a header line "N_words N_dims". GloVe and
#' word2vec .txt files start directly with word vectors. This helper
#' distinguishes the two formats by checking if the first line has exactly
#' two numeric tokens.
#'
#' @param first_line Character string, the first line of the embeddings file
#'
#' @return List with:
#'   \itemize{
#'     \item is_header: TRUE if the line is a header
#'     \item n_dims: embedding dimensionality (from header or inferred from
#'       the number of space-separated tokens minus one)
#'     \item parts: character vector of space-split tokens (reusable by caller
#'       to avoid re-splitting)
#'   }
#'
#' @keywords internal
.parse_embeddings_header <- function(first_line) {
  parts <- strsplit(first_line, " ")[[1]]
  is_header <- !is.na(suppressWarnings(as.numeric(parts[1]))) &&
               length(parts) == 2
  list(
    is_header = is_header,
    n_dims    = if (is_header) as.integer(parts[2]) else length(parts) - 1L,
    parts     = parts
  )
}


#' Normalize a numeric vector to unit Euclidean norm
#'
#' @description
#' Divides the vector by its L2 norm. Zero-norm vectors are returned unchanged
#' to avoid division by zero.
#'
#' @param v Numeric vector
#' @return Numeric vector with ||v|| = 1 (or unchanged if input was zero)
#'
#' @keywords internal
.unit_norm <- function(v) {
  n <- sqrt(sum(v^2))
  if (n > 0) v / n else v
}


#' Load word vectors from embeddings file
#'
#' @description
#' Streams through a word embeddings file and extracts vectors for specified words.
#' Supports FastText .vec, word2vec .txt, and GloVe .txt formats.
#'
#' @param seed_words Character vector of words to extract
#' @param embeddings_path Path to embeddings file
#' @param verbose Logical, print progress messages
#'
#' @return Named list where each element is a numeric vector (the word embedding)
#'
#' @keywords internal
.load_seed_vectors <- function(seed_words, embeddings_path, verbose = TRUE) {

  # Build a lookup set for O(1) membership tests. Case-insensitive matching:
  # all seed words are lowercased, and each line's word is lowercased before
  # checking membership.
  seed_words_lower <- tolower(seed_words)
  seed_words_set <- unique(seed_words_lower)
  n_needed <- length(seed_words_set)

  # Storage: named list mapping word → numeric vector (embedding)
  word_vectors <- list()

  # Stream through the file line-by-line in chunks. This avoids loading the
  # entire embeddings file (~2 GB) into memory when we only need a few
  # hundred dictionary words.
  con <- file(embeddings_path, "r", encoding = "UTF-8")
  on.exit(close(con))

  if (verbose) {
    cli_alert_info("Loading {n_needed} word vectors from embeddings...")
  }

  # Detect whether the first line is a FastText-style header ("N_words N_dims")
  # or an actual word vector (GloVe/word2vec format). If it's data, process it.
  first_line <- readLines(con, n = 1, warn = FALSE)
  header <- .parse_embeddings_header(first_line)

  # If not a header, the first line is a word vector — process it
  if (!header$is_header && length(header$parts) > 2) {
    word <- tolower(header$parts[1])
    if (word %in% seed_words_set) {
      vector_vals <- as.numeric(header$parts[-1])
      if (!any(is.na(vector_vals))) {
        word_vectors[[word]] <- vector_vals
      }
    }
  }

  # Process remaining lines in chunks of 10k for I/O efficiency.
  # Early termination: stop as soon as all needed words are found.
  lines_read <- 0
  chunk_size <- 10000

  repeat {
    lines <- readLines(con, n = chunk_size, warn = FALSE)

    if (length(lines) == 0) {
      break  # End of file
    }

    lines_read <- lines_read + length(lines)

    # Process each line
    for (line in lines) {
      # Skip empty lines
      if (nchar(trimws(line)) == 0) next

      # Split line into word and vector components
      parts <- strsplit(line, " ", fixed = TRUE)[[1]]

      if (length(parts) < 3) next  # Need at least word + 2 dimensions

      word <- tolower(parts[1])

      # Check if this is a word we need
      if (word %in% seed_words_set) {
        # Convert vector components to numeric
        vector_vals <- suppressWarnings(as.numeric(parts[-1]))

        # Only add if parsing was successful (no NAs)
        if (!any(is.na(vector_vals))) {
          word_vectors[[word]] <- vector_vals

          # Early termination if we found all words
          if (length(word_vectors) >= n_needed) {
            if (verbose) {
              cli_alert_success("Found all {n_needed} words after {lines_read} lines")
            }
            return(word_vectors)
          }
        }
      }
    }

    # Progress update
    if (verbose && lines_read %% 100000 == 0) {
      cli_alert_info("Scanned {lines_read} lines, found {length(word_vectors)}/{n_needed} words")
    }
  }

  if (verbose) {
    cli_alert_info("Found {length(word_vectors)}/{n_needed} words in embeddings")
    if (length(word_vectors) < n_needed) {
      missing_words <- setdiff(seed_words_set, names(word_vectors))
      cli_alert_warning("Missing {length(missing_words)} words from embeddings")
    }
  }

  return(word_vectors)
}


#' Get or download a hunspell dictionary for any language
#'
#' @description
#' Returns a hunspell dictionary object for the given language code. If the
#' dictionary is not installed locally, downloads the .dic and .aff files from
#' the wooorm/dictionaries GitHub repository and caches them in vdic_data/dictionaries/.
#'
#' @param language Language code (e.g., "en", "pt", "es", "fr", "de").
#'   Must match a directory name in https://github.com/wooorm/dictionaries
#' @param verbose Logical, print progress messages
#'
#' @return A hunspell dictionary object
#'
#' @keywords internal
.get_hunspell_dict <- function(language, verbose = TRUE) {

  # Cache directory: vdic_data/dictionaries/{language}/ in working directory
  cache_dir <- file.path(getwd(), "vdic_data", "dictionaries", language)
  dic_path <- file.path(cache_dir, "index.dic")
  aff_path <- file.path(cache_dir, "index.aff")

  # If already cached, load and return

  if (file.exists(dic_path) && file.exists(aff_path)) {
    if (verbose) cli_alert_info("Using cached hunspell dictionary for '{language}'")
    return(hunspell::dictionary(dic_path))
  }

  # Download from wooorm/dictionaries
  base_url <- "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries"
  dic_url <- sprintf("%s/%s/index.dic", base_url, language)
  aff_url <- sprintf("%s/%s/index.aff", base_url, language)

  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

  if (verbose) {
    cli_alert_info("Downloading hunspell dictionary for '{language}'...")
  }

  # Download .dic file
  dic_result <- tryCatch(
    {
      download.file(dic_url, dic_path, mode = "wb", quiet = TRUE)
      TRUE
    },
    error = function(e) FALSE
  )

  # Check for 404 (download.file may "succeed" but produce a small HTML error page)
  if (!dic_result || !file.exists(dic_path) || file.info(dic_path)$size < 100) {
    unlink(cache_dir, recursive = TRUE)
    stop(
      "Hunspell dictionary not found for language '", language, "'.\n",
      "Check available languages at: https://github.com/wooorm/dictionaries\n",
      "Common codes: en, pt, es, fr, de, it, nl, ru, zh, ja, ko"
    )
  }

  # Download .aff file
  aff_result <- tryCatch(
    {
      download.file(aff_url, aff_path, mode = "wb", quiet = TRUE)
      TRUE
    },
    error = function(e) FALSE
  )

  if (!aff_result || !file.exists(aff_path) || file.info(aff_path)$size < 100) {
    unlink(cache_dir, recursive = TRUE)
    stop(
      "Hunspell .aff file not found for language '", language, "'.\n",
      "The dictionary may be incomplete at: https://github.com/wooorm/dictionaries"
    )
  }

  if (verbose) {
    cli_alert_success("Downloaded hunspell dictionary for '{language}' to {.path {cache_dir}}")
  }

  hunspell::dictionary(dic_path)
}


#' Filter embedding vocabulary by spellcheck and stopwords
#'
#' @description
#' Reads the full embeddings file and filters out stopwords and non-words
#' (via hunspell spell checking). Writes a temporary filtered embeddings file
#' that downstream functions can use directly, so no other function needs
#' spellcheck/stopword logic.
#'
#' @param embeddings_path Path to embeddings file
#' @param stopwords Character vector of stopwords to remove, or NULL
#' @param spellcheck Logical. If TRUE, filters using hunspell spell checking
#' @param language Language code for hunspell dictionary (e.g., "en", "pt", "es").
#'   If NULL, defaults to "en". Used to select the correct hunspell dictionary
#'   so that spell checking works for non-English embeddings.
#' @param verbose Logical, print progress messages
#'
#' @return Path to filtered embeddings file (temp file, or original if no filtering)
#'
#' @keywords internal
.filter_embedding_vocab <- function(embeddings_path, stopwords = NULL, spellcheck = FALSE, language = NULL, verbose = TRUE) {

  # ── Determine which filters to apply ──
  # Non-alphabetic token removal always runs (catches number codes, URLs, etc.).
  # Stopword removal and spellcheck are optional.
  use_stopwords <- !is.null(stopwords) && length(stopwords) > 0
  use_spellcheck <- isTRUE(spellcheck)

  # Spellcheck requires hunspell
  if (use_spellcheck && !requireNamespace("hunspell", quietly = TRUE)) {
    stop("Package 'hunspell' is required for spellcheck = TRUE. ",
         "Install it with: install.packages('hunspell')")
  }

  # Resolve hunspell dictionary from language code
  hunspell_dict <- NULL
  if (use_spellcheck) {
    hunspell_dict <- .get_hunspell_dict(language, verbose)
  }

  # Two code paths: data.table (fast, in-memory) vs base R (streaming, low memory).
  # data.table loads the entire file at once (~2 GB), which is much faster but
  # requires enough RAM. The streaming fallback processes line-by-line.
  has_dt <- requireNamespace("data.table", quietly = TRUE)

  if (has_dt) {
    # ── Fast path: data.table (loads entire file into memory) ──
    header <- .parse_embeddings_header(readLines(embeddings_path, n = 1, warn = FALSE))
    n_dims <- header$n_dims
    skip_rows <- if (header$is_header) 1L else 0L

    if (verbose) {
      cli_alert_info("Reading embeddings ({n_dims} dimensions)...")
    }

    dt <- data.table::fread(
      embeddings_path,
      header = FALSE,
      skip = skip_rows,
      sep = " ",
      quote = "",
      colClasses = c("character", rep("numeric", n_dims)),
      showProgress = FALSE,
      data.table = TRUE
    )

    n_total <- nrow(dt)
    if (verbose) {
      cli_alert_info("Read {n_total} words")
    }

    # Lowercase all words for case-insensitive matching throughout the pipeline
    words <- tolower(dt[[1]])

    # Remove duplicate words arising from case variants (e.g., "Gás" and "gás"
    # both become "gás"). Keep first occurrence — FastText orders by frequency,
    # so the first casing variant is typically the most common and best-quality vector.
    dup_mask <- duplicated(words)
    n_dups <- sum(dup_mask)
    if (n_dups > 0) {
      dt <- dt[!dup_mask, ]
      words <- words[!dup_mask]
      if (verbose) {
        cli_alert_info("Removed {n_dups} duplicate words (case variants)")
      }
    }

    # Remove tokens that are not purely alphabetic. The Unicode-aware regex
    # \p{L} matches any letter in any script (accented: é, ã, ü; Cyrillic: я;
    # CJK characters, etc.) while rejecting digits, punctuation, and mixed tokens
    # like "355030801-477-004879-1-0" common in Common Crawl embeddings.
    is_alpha <- grepl("^\\p{L}+$", words, perl = TRUE)
    n_non_alpha <- sum(!is_alpha)
    if (n_non_alpha > 0) {
      dt <- dt[is_alpha, ]
      words <- words[is_alpha]
      if (verbose) {
        cli_alert_info("Removed {n_non_alpha} non-alphabetic tokens (numbers, codes, symbols)")
      }
    }

    # Remove stopwords
    if (use_stopwords) {
      stopwords_lower <- tolower(stopwords)
      keep_mask <- !words %in% stopwords_lower
      n_removed <- sum(!keep_mask)
      dt <- dt[keep_mask, ]
      words <- words[keep_mask]
      if (verbose) {
        cli_alert_info("Removed {n_removed} stopwords, {nrow(dt)} words remaining")
      }
    }

    # Apply spellcheck
    if (use_spellcheck) {
      if (verbose) {
        cli_alert_info("Spell checking {length(words)} words")
      }
      n_before <- length(words)
      valid_words <- hunspell::hunspell_check(words, dict = hunspell_dict)
      dt <- dt[valid_words, ]
      words <- words[valid_words]

      if (verbose) {
        n_removed <- n_before - length(words)
        cli_alert_info("Removed {n_removed} non-words, {length(words)} valid words remaining")
      }
    }

    # Write filtered file to temp location
    filtered_path <- tempfile(fileext = ".vec")

    if (verbose) {
      cli_alert_info("Writing filtered embeddings to temp file...")
    }

    # Set first column to lowercased words
    data.table::set(dt, j = 1L, value = words)

    # Build output lines as character vectors and write with writeLines().
    # We avoid data.table::fwrite() because words that look like numbers
    # (e.g., "46") cause fread() to misparse the line on re-read (field
    # count mismatch when it tries to auto-detect column types).
    embed_cols <- as.matrix(dt[, 2:(n_dims + 1), with = FALSE])
    vec_strings <- apply(embed_cols, 1, function(row) paste(row, collapse = " "))
    lines_out <- paste(words, vec_strings)

    writeLines(c(paste(length(lines_out), n_dims), lines_out), filtered_path)

    if (verbose) {
      cli_alert_success("Filtered vocabulary: {n_total} -> {length(words)} words")
    }

  } else {
    # ── Streaming fallback (base R) ──
    if (use_spellcheck) {
      stop("Package 'data.table' is required for spellcheck = TRUE. ",
           "Install it with: install.packages('data.table')")
    }

    stopwords_set <- if (use_stopwords) tolower(stopwords) else character(0)

    # Strategy: write the filtered word vectors to a body temp file (no header),
    # then create the final file by writing the header line + copying the body.
    # This avoids needing a second pass over the entire file just to count lines
    # for the header, since we don't know the final word count until streaming ends.
    body_path <- tempfile(fileext = ".body")
    filtered_path <- tempfile(fileext = ".vec")

    con_in <- file(embeddings_path, "r", encoding = "UTF-8")
    con_body <- file(body_path, "w", encoding = "UTF-8")
    on.exit({ close(con_in); close(con_body) }, add = TRUE)

    # Parse header
    first_line <- readLines(con_in, n = 1, warn = FALSE)
    header <- .parse_embeddings_header(first_line)

    n_kept <- 0
    n_dims_out <- header$n_dims
    seen_words <- character(0)  # Track seen words for case-variant deduplication

    # Process first line if not a header
    if (!header$is_header && length(header$parts) > 2) {
      word <- tolower(header$parts[1])
      if (grepl("^\\p{L}+$", word, perl = TRUE) && !(word %in% stopwords_set)) {
        header$parts[1] <- word
        writeLines(paste(header$parts, collapse = " "), con_body)
        seen_words <- c(seen_words, word)
        n_kept <- n_kept + 1
      }
    }

    # Stream through file
    chunk_size <- 50000
    lines_read <- 0

    if (verbose) {
      cli_alert_info("Streaming and filtering embeddings (stopwords only, no spellcheck)...")
    }

    repeat {
      lines <- readLines(con_in, n = chunk_size, warn = FALSE)
      if (length(lines) == 0) break
      lines_read <- lines_read + length(lines)

      for (line in lines) {
        if (nchar(trimws(line)) == 0) next
        space_pos <- regexpr(" ", line, fixed = TRUE)
        if (space_pos < 1) next
        word <- tolower(substr(line, 1, space_pos - 1))
        if (!grepl("^\\p{L}+$", word, perl = TRUE)) next
        if (word %in% stopwords_set) next
        if (word %in% seen_words) next  # Deduplicate case variants

        # Rewrite with lowercase word
        rest <- substr(line, space_pos, nchar(line))
        writeLines(paste0(word, rest), con_body)
        seen_words <- c(seen_words, word)
        n_kept <- n_kept + 1
      }

      if (verbose && lines_read %% 500000 == 0) {
        cli_alert_info("Processed {lines_read} lines, kept {n_kept} words")
      }
    }

    # Close connections (the on.exit handler becomes a harmless no-op)
    close(con_in)
    close(con_body)

    # Assemble final file: write the FastText-style header ("N_words N_dims"),
    # then copy the body file contents in chunks. This produces a valid .vec
    # file that downstream functions can parse with .parse_embeddings_header().
    con_out <- file(filtered_path, "w", encoding = "UTF-8")
    writeLines(paste(n_kept, n_dims_out), con_out)

    con_body_in <- file(body_path, "r", encoding = "UTF-8")
    while (length(chunk <- readLines(con_body_in, n = chunk_size, warn = FALSE)) > 0) {
      writeLines(chunk, con_out)
    }
    close(con_body_in)
    close(con_out)
    unlink(body_path)

    if (verbose) {
      cli_alert_success("Filtered vocabulary: {lines_read} -> {n_kept} words (stopwords removed)")
    }
  }

  return(filtered_path)
}


#' Project all embeddings onto learned axes
#'
#' @description
#' Reads the embeddings file and computes projections for ALL words onto the
#' learned axes. Uses data.table::fread() for fast parsing when available,
#' with fallback to base R streaming for large files.
#'
#' This replicates the Duan et al. (2025) approach where the vectionary can
#' score any word in the vocabulary.
#'
#' @param axes Named list of axis vectors (one per dimension)
#' @param dimensions Character vector of dimension names
#' @param embeddings_path Path to embeddings file (should already be filtered)
#' @param verbose Logical, print progress messages
#'
#' @return Data frame with 'word' column and one column per dimension containing projections
#'
#' @keywords internal
.project_all_embeddings <- function(axes, dimensions, embeddings_path, verbose = TRUE) {

  # Dispatch to fast (data.table) or streaming (base R) implementation.
  # Both produce identical results; the choice is purely about performance.
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(.project_all_embeddings_fast(axes, dimensions, embeddings_path, verbose))
  }
  return(.project_all_embeddings_streaming(axes, dimensions, embeddings_path, verbose))
}


#' Fast projection using data.table::fread (5-10x faster)
#'
#' @keywords internal
.project_all_embeddings_fast <- function(axes, dimensions, embeddings_path, verbose = TRUE) {

  header <- .parse_embeddings_header(readLines(embeddings_path, n = 1, warn = FALSE))
  n_dims <- header$n_dims
  skip_rows <- if (header$is_header) 1L else 0L

  if (verbose) {
    cli_alert_info("Reading embeddings with {n_dims} dimensions...")
  }

  dt <- data.table::fread(
    embeddings_path,
    header = FALSE,
    skip = skip_rows,
    sep = " ",
    quote = "",
    colClasses = c("character", rep("numeric", n_dims)),
    showProgress = FALSE,
    data.table = TRUE
  )

  # Words are already lowercase and deduplicated by .filter_embedding_vocab(),
  # so no further cleaning is needed here.
  words <- dt[[1]]

  if (verbose) {
    cli_alert_info("Read {length(words)} words, computing projections...")
  }

  # Convert to matrix for vectorized operations.
  # Shape: (n_words x embedding_dims), e.g., (500000 x 300)
  embed_matrix <- as.matrix(dt[, 2:(n_dims + 1), with = FALSE])

  # Normalize each word vector to unit Euclidean norm. This MUST match the
  # normalization applied to seed words during axis learning (.unit_norm),
  # so that projections (dot products) are cosine-similarity-based.
  # Without this, frequent words in FastText have inflated norms that would
  # dominate dot products regardless of semantic content.
  row_norms <- sqrt(rowSums(embed_matrix^2))
  row_norms[row_norms == 0] <- 1  # guard against zero-norm vectors
  embed_matrix <- embed_matrix / row_norms

  # Stack axes into a matrix. Shape: (embedding_dims x n_axes), e.g., (300 x 5)
  axis_matrix <- do.call(cbind, axes[dimensions])

  # Single matrix multiply projects all words onto all axes at once.
  # Result shape: (n_words x n_axes). Each cell is the dot product of a
  # unit-norm word embedding with a (non-unit-norm) learned axis.
  proj_matrix <- embed_matrix %*% axis_matrix

  if (verbose) {
    cli_alert_success("Projected {length(words)} words onto {length(dimensions)} axes")
  }

  # Build result data frame
  result <- data.frame(word = words, stringsAsFactors = FALSE)
  for (i in seq_along(dimensions)) {
    result[[dimensions[i]]] <- proj_matrix[, i]
  }

  return(result)
}


#' Streaming projection fallback (for systems without data.table)
#'
#' @keywords internal
.project_all_embeddings_streaming <- function(axes, dimensions, embeddings_path, verbose = TRUE) {

  if (verbose) {
    cli_alert_info("Using streaming projection (install data.table for 5-10x speedup)")
  }

  # ── Chunk-based collection strategy ──
  # Collect results as lists of chunks, then combine once at the end with
  # unlist() / do.call(rbind, ...). This is O(n) total cost.
  # The naive alternative — growing vectors with c() on each line — is O(n^2)
  # because R copies the entire vector on each append.
  words_list <- list()   # list of character vectors (one per chunk)
  proj_list  <- list()   # list of projection matrices (one per chunk)
  chunk_idx  <- 0L

  # Open file connection
  con <- file(embeddings_path, "r", encoding = "UTF-8")
  on.exit(close(con))

  # Parse header
  first_line <- readLines(con, n = 1, warn = FALSE)
  header <- .parse_embeddings_header(first_line)

  if (header$is_header) {
    if (verbose) {
      cli_alert_info("Embeddings: {header$parts[1]} words, {header$n_dims} dimensions")
    }
  } else {
    # First line is a word vector — process it
    if (length(header$parts) > 2) {
      vector_vals <- as.numeric(header$parts[-1])
      if (!any(is.na(vector_vals))) {
        vector_vals <- .unit_norm(vector_vals)
        chunk_idx <- chunk_idx + 1L
        words_list[[chunk_idx]] <- header$parts[1]  # already lowercase from filter
        proj_list[[chunk_idx]] <- matrix(vector_vals, nrow = 1) %*%
          do.call(cbind, axes[dimensions])
      }
    }
  }

  # Build axis matrix for faster projection
  axis_matrix <- do.call(cbind, axes[dimensions])

  # Process remaining lines in chunks
  lines_read <- 0
  chunk_size <- 50000
  words_processed <- if (chunk_idx > 0) 1L else 0L

  if (verbose) {
    cli_alert_info("Projecting all embeddings onto {length(dimensions)} axes...")
  }

  repeat {
    lines <- readLines(con, n = chunk_size, warn = FALSE)
    if (length(lines) == 0) break

    lines_read <- lines_read + length(lines)

    # Pre-allocate for this chunk
    chunk_words <- character(length(lines))
    chunk_vectors <- vector("list", length(lines))
    valid_count <- 0

    for (i in seq_along(lines)) {
      line <- lines[i]
      if (nchar(trimws(line)) == 0) next

      parts <- strsplit(line, " ", fixed = TRUE)[[1]]
      if (length(parts) < 3) next

      vector_vals <- suppressWarnings(as.numeric(parts[-1]))

      if (!any(is.na(vector_vals))) {
        valid_count <- valid_count + 1
        chunk_words[valid_count] <- parts[1]  # already lowercase from filter
        chunk_vectors[[valid_count]] <- vector_vals
      }
    }

    # Vectorized projection for entire chunk at once (matrix multiply)
    if (valid_count > 0) {
      chunk_words <- chunk_words[1:valid_count]
      chunk_matrix <- do.call(rbind, chunk_vectors[1:valid_count])

      # Unit-normalize to match axis learning normalization
      row_norms <- sqrt(rowSums(chunk_matrix^2))
      row_norms[row_norms == 0] <- 1
      chunk_matrix <- chunk_matrix / row_norms

      # Store this chunk's results (combined later)
      chunk_idx <- chunk_idx + 1L
      words_list[[chunk_idx]] <- chunk_words
      proj_list[[chunk_idx]]  <- chunk_matrix %*% axis_matrix
      words_processed <- words_processed + valid_count
    }

    if (verbose && lines_read %% 500000 == 0) {
      cli_alert_info("Processed {lines_read} lines, {words_processed} words projected")
    }
  }

  if (verbose) {
    cli_alert_success("Projected {words_processed} words onto all axes")
  }

  # Combine all chunks into final vectors/matrix in a single pass
  words <- unlist(words_list, use.names = FALSE)
  proj_matrix <- do.call(rbind, proj_list)  # stacks chunk matrices vertically

  result <- data.frame(word = words, stringsAsFactors = FALSE)
  for (i in seq_along(dimensions)) {
    result[[dimensions[i]]] <- proj_matrix[, i]
  }

  return(result)
}


#' Expand stem patterns in dictionary
#'
#' @description
#' Scans embeddings file for words matching stem patterns (e.g., "abandon*")
#' and expands them to all matching full words. Each expanded word inherits
#' the scores from its stem pattern.
#'
#' @param dictionary Data frame with 'word' column and dimension columns.
#'   Words ending with * are treated as stem patterns.
#' @param embeddings_path Path to embeddings file (should already be filtered)
#' @param verbose Print progress messages
#'
#' @return Expanded dictionary with stem patterns replaced by matching words
#'
#' @keywords internal
.expand_stems <- function(dictionary, embeddings_path, verbose = TRUE) {

  # Separate the dictionary into two groups:
  #   stems: words ending with * (e.g., "abandon*") → will be expanded
  #   exact: regular words → kept as-is
  is_stem <- grepl("\\*$", dictionary$word)
  stems <- dictionary[is_stem, ]
  exact <- dictionary[!is_stem, ]

  if (nrow(stems) == 0) {
    return(dictionary)
  }

  # Convert stem patterns to anchored regexes: "abandon*" → "^abandon"
  # This matches any word starting with the stem (abandon, abandoned, etc.)
  stem_patterns <- sub("\\*$", "", tolower(stems$word))
  stem_regex <- paste0("^", stem_patterns)

  # Each expanded word inherits the scores from its matching stem pattern.
  # Collected as a list of data frame rows, then rbind'd at the end.
  expanded_rows <- list()

  # Open embeddings file
  con <- file(embeddings_path, "r", encoding = "UTF-8")
  on.exit(close(con))

  # Parse header
  first_line <- readLines(con, n = 1, warn = FALSE)
  header <- .parse_embeddings_header(first_line)

  # Helper to check if a word matches any stem and return matching row
  match_stem <- function(word) {
    for (i in seq_along(stem_regex)) {
      if (grepl(stem_regex[i], word)) {
        return(i)
      }
    }
    return(NA)
  }

  # Process first line if not a header
  if (!header$is_header && length(header$parts) > 1) {
    word <- tolower(header$parts[1])
    stem_idx <- match_stem(word)
    if (!is.na(stem_idx)) {
      # Create new row with this word and scores from matching stem
      new_row <- stems[stem_idx, ]
      new_row$word <- word
      expanded_rows[[length(expanded_rows) + 1]] <- new_row
    }
  }

  # Stream through embeddings
  lines_read <- 0
  chunk_size <- 50000

  if (verbose) {
    cli_progress_step("Scanning embeddings for stem matches", spinner = TRUE)
  }

  repeat {
    lines <- readLines(con, n = chunk_size, warn = FALSE)
    if (length(lines) == 0) break

    lines_read <- lines_read + length(lines)

    for (line in lines) {
      if (nchar(trimws(line)) == 0) next

      # Extract just the word (first element before space)
      space_pos <- regexpr(" ", line, fixed = TRUE)
      if (space_pos < 1) next

      word <- tolower(substr(line, 1, space_pos - 1))

      # Check if matches any stem pattern
      stem_idx <- match_stem(word)
      if (!is.na(stem_idx)) {
        # Create new row with this word and scores from matching stem
        new_row <- stems[stem_idx, ]
        new_row$word <- word
        expanded_rows[[length(expanded_rows) + 1]] <- new_row
      }
    }

    if (verbose && lines_read %% 500000 == 0) {
      cli_progress_step("Scanned {lines_read} lines, found {length(expanded_rows)} matches", spinner = TRUE)
    }
  }

  if (verbose) {
    cli_progress_done()
  }

  # Combine expanded words with exact words
  if (length(expanded_rows) > 0) {
    expanded_df <- do.call(rbind, expanded_rows)

    # Remove duplicates (keep first occurrence)
    expanded_df <- expanded_df[!duplicated(expanded_df$word), ]

    # Also remove any that conflict with exact words
    expanded_df <- expanded_df[!expanded_df$word %in% tolower(exact$word), ]

    if (verbose) {
      cli_alert_info("Expanded {nrow(stems)} stem{?s} to {nrow(expanded_df)} word{?s}")
    }

    # Combine with exact words
    result <- rbind(exact, expanded_df)
  } else {
    if (verbose) {
      cli_alert_warning("No words found matching stem patterns")
    }
    result <- exact
  }

  rownames(result) <- NULL
  return(result)
}


#' Sample random word vectors from embeddings file
#'
#' @description
#' Samples random words from an embeddings file, excluding specified words.
#' Used for AUC validation of binary dictionaries.
#'
#' @param embeddings_path Path to embeddings file
#' @param n_sample Number of random words to sample
#' @param exclude_words Words to exclude (e.g., dictionary words)
#' @param seed Random seed for reproducibility
#'
#' @return Named list of word vectors
#'
#' @keywords internal
.sample_random_vectors <- function(embeddings_path, n_sample = 500, exclude_words = NULL, seed) {

  # ── RNG state preservation ──
  # Save the global .Random.seed before calling set.seed(), then restore it
  # on exit. This ensures that random sampling inside this function does not
  # alter the caller's RNG stream, which matters when vectionary_builder()
  # runs multiple steps that each need deterministic randomness.
  old_seed <- if (exists(".Random.seed", envir = globalenv())) {
    get(".Random.seed", envir = globalenv())
  } else NULL
  on.exit({
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = globalenv())) rm(".Random.seed", envir = globalenv())
    } else {
      assign(".Random.seed", old_seed, envir = globalenv())
    }
  })
  set.seed(seed)

  exclude_set <- if (!is.null(exclude_words)) tolower(exclude_words) else character(0)

  # Open file and parse header
  con <- file(embeddings_path, "r", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)

  first_line <- readLines(con, n = 1, warn = FALSE)
  header <- .parse_embeddings_header(first_line)

  # Collect 3x more candidates than needed to have a good pool after filtering
  # out excluded words. The final sample is drawn from this pool.
  candidates <- list()
  lines_read <- 0
  target_candidates <- n_sample * 3

  # If not header, process first line
  if (!header$is_header && length(header$parts) > 2) {
    word <- tolower(header$parts[1])
    if (!(word %in% exclude_set) && grepl("^\\p{L}+$", word, perl = TRUE)) {
      vector_vals <- as.numeric(header$parts[-1])
      if (!any(is.na(vector_vals))) {
        candidates[[word]] <- vector_vals
      }
    }
  }

  # Sample by reading chunks and randomly selecting
  chunk_size <- 50000
  repeat {
    lines <- readLines(con, n = chunk_size, warn = FALSE)
    if (length(lines) == 0) break

    lines_read <- lines_read + length(lines)

    # Randomly sample lines from this chunk
    sample_idx <- sample(seq_along(lines), min(500, length(lines)))

    for (i in sample_idx) {
      line <- lines[i]
      if (nchar(trimws(line)) == 0) next

      parts <- strsplit(line, " ", fixed = TRUE)[[1]]
      if (length(parts) < 3) next

      word <- tolower(parts[1])

      # Skip excluded words and non-alphabetic words (Unicode-aware)
      if (word %in% exclude_set) next
      if (!grepl("^\\p{L}+$", word, perl = TRUE)) next

      vector_vals <- suppressWarnings(as.numeric(parts[-1]))
      if (!any(is.na(vector_vals))) {
        candidates[[word]] <- vector_vals
      }

      if (length(candidates) >= target_candidates) break
    }

    if (length(candidates) >= target_candidates) break
  }

  # Subsample to exactly n_sample from the candidate pool
  if (length(candidates) > n_sample) {
    sample_names <- sample(names(candidates), n_sample)
    candidates <- candidates[sample_names]
  }

  # Unit-normalize to match the normalization applied during axis learning,
  # so that AUC projections are comparable to dictionary word projections.
  candidates <- lapply(candidates, .unit_norm)

  return(candidates)
}


#' Expand vocabulary with semantically related words
#'
#' @description
#' Finds the top-N words from the already-computed word projections that project
#' most strongly onto the learned axes. Uses the projection results from a
#' previous build step, so no file I/O is needed.
#'
#' @param word_projections Data frame with 'word' column and dimension columns
#'   (from .build_vectionary_internal() result)
#' @param dictionary Data frame with 'word' column (to exclude from expansion)
#' @param dimensions Character vector of dimension names
#' @param n_expand Number of words to add
#' @param positive_only If TRUE, only add words with positive average projection
#' @param verbose Print progress
#'
#' @return Data frame with word and projection columns for each dimension
#'
#' @keywords internal
.expand_vocabulary <- function(word_projections, dictionary, dimensions, n_expand, positive_only = TRUE, verbose = TRUE) {

  exclude_set <- tolower(dictionary$word)

  # Filter out existing dictionary words and tokens shorter than 3 letters.
  # The Unicode regex \p{L}{3,} ensures only real words (no abbreviations or
  # single-letter tokens) are considered for expansion.
  candidates <- word_projections[!tolower(word_projections$word) %in% exclude_set, ]
  candidates <- candidates[grepl("^\\p{L}{3,}$", candidates$word, perl = TRUE), ]

  if (nrow(candidates) == 0) {
    if (verbose) {
      cli_alert_warning("No candidate words found for vocabulary expansion")
    }
    return(data.frame(word = character(0), stringsAsFactors = FALSE))
  }

  # Rank candidates by average projection strength across all dimensions.
  # For binary dictionaries (positive_only=TRUE), only words with positive
  # average projection are useful — negative projections indicate the word
  # is semantically opposite to the dictionary concept.
  candidates$avg_proj <- rowMeans(candidates[, dimensions, drop = FALSE])
  if (positive_only) {
    candidates <- candidates[candidates$avg_proj > 0, ]
  }

  if (nrow(candidates) == 0) {
    if (verbose) {
      cli_alert_warning("No words with positive projections found for expansion")
    }
    return(data.frame(word = character(0), stringsAsFactors = FALSE))
  }

  # Sort by absolute projection value (keeps strongest matches)
  candidates <- candidates[order(abs(candidates$avg_proj), decreasing = TRUE), ]

  # Take top n_expand
  expanded_df <- head(candidates, n_expand)
  expanded_df$avg_proj <- NULL

  rownames(expanded_df) <- NULL

  if (verbose) {
    cli_alert_success("Found {nrow(expanded_df)} words for vocabulary expansion")
  }

  return(expanded_df)
}


#' Solve for optimal axis using ridge regression
#'
#' @description
#' Finds the optimal axis in embedding space for a dimension using ridge regression.
#' Implements: a = (W^T W + lambda I)^-1 W^T y
#'
#' @param word_vectors List of numeric vectors (embeddings)
#' @param word_scores Numeric vector of scores for each word
#' @param lambda Ridge regularization parameter (default: 1.0)
#'
#' @return Numeric vector representing the axis. Not unit-normalized; scale encodes
#'   dictionary scores when used with unit-norm embeddings.
#'
#' @keywords internal
.solve_axis_ridge <- function(word_vectors, word_scores, lambda = 1.0) {

  # W: word embedding matrix (n_words x embedding_dims), e.g., (20 x 300)
  # y: target scores for each word (length n_words)
  W <- do.call(rbind, word_vectors)
  y <- as.numeric(word_scores)

  # ── Ridge regression closed-form solution ──
  # $a = (W^T W + \lambda I)^{-1} W^T y$
  #
  # The L2 penalty $\lambda I$ shrinks the axis toward zero, preventing
  # overfitting when embedding_dims >> n_words. Higher lambda = more shrinkage.
  # The axis is NOT unit-normalized — its magnitude encodes the dictionary
  # score scale, so seed words project to approximately their original scores.

  # crossprod(W) is faster than t(W) %*% W (avoids explicit transpose)
  WtW <- crossprod(W)

  # Add ridge penalty to the diagonal
  WtW <- WtW + lambda * diag(ncol(W))

  # Right-hand side of the normal equations
  Wty <- crossprod(W, y)

  # Solve the linear system. If the matrix is still singular (rare, usually
  # when lambda ≈ 0 and n_words << embedding_dims), fall back to pseudo-inverse.
  axis <- tryCatch(
    solve(WtW, Wty),
    error = function(e) {
      cli_alert_warning("Embeddings have {ncol(W)} dimensions, dictionary has {nrow(W)} words. Singular matrix, using pseudo-inverse.")
      MASS::ginv(WtW) %*% Wty
    }
  )

  return(as.numeric(axis))
}


#' Solve for axis using Elastic Net regression
#'
#' @description
#' Learns a semantic axis using elastic net regularization (combination of L1 and L2 penalties).
#' Elastic net encourages sparsity (L1) while handling correlated features (L2).
#'
#' @param word_vectors List of embedding vectors for dictionary words
#' @param word_scores Numeric vector of scores for words on this dimension
#' @param lambda Regularization strength (higher = more regularization)
#' @param l1_ratio Mixing parameter between L1 and L2 (0 = ridge, 1 = lasso, 0.5 = equal mix)
#'
#' @return Numeric vector representing the learned axis. Not unit-normalized; scale
#'   encodes dictionary scores when used with unit-norm embeddings.
#'
#' @keywords internal
.solve_axis_elastic_net <- function(word_vectors, word_scores, lambda = 1.0, l1_ratio = 0.5) {

  # W: word embedding matrix (n_words x embedding_dims)
  # y: target scores for each word
  W <- do.call(rbind, word_vectors)
  y <- as.numeric(word_scores)

  # ── Elastic net via glmnet ──
  # glmnet minimizes: (1/2n) ||y - Wa||^2 + lambda * [alpha * ||a||_1 + (1-alpha)/2 * ||a||_2^2]
  #
  # NAMING: our "lambda" = glmnet's "lambda" (overall regularization strength)
  #         our "l1_ratio" = glmnet's "alpha" (L1 vs L2 mixing)
  # standardize=FALSE: embeddings are already unit-normalized
  # intercept=FALSE: axes must pass through the origin (no bias term)
  fit <- glmnet::glmnet(
    x = W,
    y = y,
    alpha = l1_ratio,
    lambda = lambda,
    standardize = FALSE,
    intercept = FALSE
  )

  # Extract axis coefficients (first element of coef() is the intercept = 0)
  axis <- as.numeric(coef(fit, s = lambda))[-1]

  # If lambda is too large, the L1 penalty can shrink ALL coefficients to zero.
  # Fall back to ridge regression which always produces a nonzero axis.
  axis_norm <- sqrt(sum(axis^2))
  if (axis_norm == 0) {
    cli::cli_alert_warning("Elastic net produced zero axis (lambda too high), falling back to ridge")
    return(.solve_axis_ridge(word_vectors, word_scores, lambda))
  }

  return(as.numeric(axis))
}


#' Solve for axis using Duan et al. (2025) constrained optimization
#'
#' @description
#' Learns a semantic axis using constrained nonlinear optimization as described
#' in Duan et al. (2025). Minimizes sum of squared errors subject to unit norm
#' constraint: min sum((w_i . m - s_i)^2) s.t. ||m|| = 1
#'
#' This replicates the vMFD methodology for direct comparison. Note that this
#' method has no regularization, which may lead to overfitting in high-dimensional
#' embedding spaces.
#'
#' @param word_vectors List of embedding vectors for dictionary words
#' @param word_scores Numeric vector of scores for words on this dimension
#' @param dim_name Dimension name (combined with seed for per-dimension reproducibility)
#' @param seed Base seed passed from vectionary_builder()
#'
#' @return Numeric vector representing the learned axis (unit norm)
#'
#' @references
#' Duan, Z., et al. (2025). Constructing Vec-tionaries to Extract Message
#' Features from Texts: A Case Study of Moral Content. Political Analysis, 1-21.
#'
#' @keywords internal
.solve_axis_duan <- function(word_vectors, word_scores, dim_name = "default", seed) {

  # W: word embedding matrix (n_words x embedding_dims)
  # y: target scores for each word
  W <- do.call(rbind, word_vectors)
  y <- as.numeric(word_scores)
  n_dims <- ncol(W)

  # ── Duan et al. (2025) constrained optimization ──
  # Minimize $\sum_i (w_i \cdot m - s_i)^2$ subject to $||m|| = 1$
  #
  # Unlike ridge/elastic net, there is NO regularization. The unit-norm
  # constraint prevents the trivial solution of scaling m to infinity.
  # This replicates the vMFD methodology exactly.

  # Objective: sum of squared prediction errors
  fn <- function(m) {
    predictions <- as.numeric(W %*% m)
    sum((predictions - y)^2)
  }

  # Analytical gradient: $\nabla = 2 W^T (Wm - y)$
  gr <- function(m) {
    predictions <- as.numeric(W %*% m)
    residuals <- predictions - y
    as.numeric(2 * crossprod(W, residuals))
  }

  # Equality constraint: $||m||^2 - 1 = 0$ (enforces unit norm)
  heq <- function(m) {
    sum(m^2) - 1
  }

  # Jacobian of constraint: $\partial h / \partial m = 2m$
  heq.jac <- function(m) {
    matrix(2 * m, nrow = 1)
  }

  # ── RNG state preservation ──
  # Same pattern as .sample_random_vectors(): save .Random.seed, set a
  # deterministic seed, and restore the original state on exit.
  old_seed <- if (exists(".Random.seed", envir = globalenv())) {
    get(".Random.seed", envir = globalenv())
  } else NULL
  on.exit({
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = globalenv())) rm(".Random.seed", envir = globalenv())
    } else {
      assign(".Random.seed", old_seed, envir = globalenv())
    }
  })

  # Generate a per-dimension seed by hashing the dimension name. This ensures
  # different dimensions get different random starting points, while the same
  # dimension always gets the same starting point for reproducibility.
  dim_seed <- sum(utf8ToInt(dim_name)) + seed
  set.seed(dim_seed)
  start <- rnorm(n_dims)
  start <- start / sqrt(sum(start^2))  # project onto unit sphere

  if (!requireNamespace("alabama", quietly = TRUE)) {
    stop("Package 'alabama' is required for method = 'duan'. ",
         "Install it with: install.packages('alabama')")
  }

  # Augmented Lagrangian method (alabama::auglag) converts the constrained
  # problem into a sequence of unconstrained subproblems using nlminb.
  result <- alabama::auglag(
    par = start,
    fn = fn,
    gr = gr,
    heq = heq,
    heq.jac = heq.jac,
    control.outer = list(trace = FALSE, method = "nlminb"),
    control.optim = list(maxit = 1000)
  )

  axis <- result$par

  # Re-normalize to exact unit norm (numerical solver may drift slightly)
  axis_norm <- sqrt(sum(axis^2))
  axis <- axis / axis_norm

  return(as.numeric(axis))
}


#' Select optimal lambda using Generalized Cross-Validation
#'
#' @description
#' Uses GCV (Golub et al., 1979) to select the optimal regularization parameter
#' for ridge regression. GCV approximates leave-one-out cross-validation with a
#' closed-form solution, making it computationally efficient.
#'
#' Based on ORF 525 (Jianqing Fan): For ridge regression with smoother matrix
#' \eqn{S = W(W'W + \lambda I)^{-1} W'}{S = W(W'W + lambda*I)^-1 W'},
#' the GCV criterion is:
#'
#' \deqn{GCV(\lambda) = \frac{n^{-1} ||y - S y||^2}{(1 - tr(S)/n)^2}}{GCV(lambda) = (RSS/n) / (1 - tr(S)/n)^2}
#'
#' The effective degrees of freedom is
#' \eqn{tr(S) = \sum_j d_j^2 / (d_j^2 + \lambda)}{tr(S) = sum(d_j^2 / (d_j^2 + lambda))}
#' where \eqn{d_j} are singular values of \eqn{W}.
#'
#' @param W Matrix of word embeddings (rows = words, cols = dimensions)
#' @param y Vector of target scores
#' @param lambda_seq Sequence of lambda values to evaluate (default: logarithmic sequence)
#' @param return_all If TRUE, return GCV values for all lambdas (for diagnostics)
#'
#' @return If return_all = FALSE: optimal lambda value
#'         If return_all = TRUE: list with optimal_lambda, lambda_seq, gcv_values, df_values
#'
#' @references
#' Golub, G. H., Heath, M., & Wahba, G. (1979). Generalized cross-validation as a
#' method for choosing a good ridge parameter. Technometrics, 21(2), 215-223.
#'
#' @keywords internal
.gcv_select_lambda <- function(W, y, lambda_seq = NULL, return_all = FALSE) {


  n <- nrow(W)  # number of seed words
  d <- ncol(W)  # embedding dimensionality (e.g., 300)

  # Search over a log-spaced grid from very weak (1e-4) to very strong (1e4)
  # regularization. 50 points gives sufficient resolution on the log scale.
  if (is.null(lambda_seq)) {
    lambda_seq <- 10^seq(-4, 4, length.out = 50)
  }

  # ── SVD-based GCV (efficient closed-form) ──
  # The SVD $W = U D V^T$ lets us express the smoother matrix as:
  #   $S_\lambda = U \cdot \text{diag}(d_j^2 / (d_j^2 + \lambda)) \cdot U^T$
  # We only need U and the singular values d (not V), saving computation.
  svd_W <- svd(W, nu = min(n, d), nv = 0)
  U <- svd_W$u
  sing_vals <- svd_W$d

  # Pre-compute $U^T y$ once — reused for every lambda candidate
  Uty <- as.numeric(crossprod(U, y))

  # Evaluate the GCV criterion for each lambda value
  results <- sapply(lambda_seq, function(lambda) {

    # Shrinkage factors: $d_j^2 / (d_j^2 + \lambda)$ ∈ [0, 1]
    # Small lambda → shrinkage ≈ 1 (nearly unregularized)
    # Large lambda → shrinkage ≈ 0 (heavily regularized)
    shrinkage <- sing_vals^2 / (sing_vals^2 + lambda)

    # Effective degrees of freedom: $\text{tr}(S_\lambda) = \sum_j d_j^2 / (d_j^2 + \lambda)$
    df_lambda <- sum(shrinkage)

    # Fitted values via SVD: $\hat{y} = U \cdot \text{diag}(\text{shrinkage}) \cdot U^T y$
    y_hat <- as.numeric(U %*% (shrinkage * Uty))

    rss <- sum((y - y_hat)^2)

    # GCV criterion: $\frac{n^{-1} \text{RSS}}{(1 - \text{tr}(S)/n)^2}$
    # Approximates leave-one-out CV without refitting n times.
    gcv <- (rss / n) / (1 - df_lambda / n)^2

    c(gcv = gcv, df = df_lambda, rss = rss)
  })

  gcv_values <- results["gcv", ]
  df_values <- results["df", ]

  # Select the lambda that minimizes the GCV criterion
  best_idx <- which.min(gcv_values)
  optimal_lambda <- lambda_seq[best_idx]

  if (return_all) {
    return(list(
      optimal_lambda = optimal_lambda,
      lambda_seq = lambda_seq,
      gcv_values = gcv_values,
      df_values = df_values,
      best_idx = best_idx
    ))
  } else {
    return(optimal_lambda)
  }
}


#' Select optimal lambda using GCV for a dictionary dimension
#'
#' @description
#' Wrapper around .gcv_select_lambda that works with the vdic dictionary format.
#' Loads word vectors and applies GCV to select optimal regularization.
#'
#' @param dictionary Data frame with 'word' column and dimension columns
#' @param dimension Name of the dimension to optimize
#' @param embeddings_path Path to embeddings file
#' @param lambda_seq Sequence of lambda values to test
#' @param verbose Print progress messages
#'
#' @return List with optimal_lambda and diagnostic information
#'
#' @keywords internal
.gcv_select_lambda_for_dimension <- function(dictionary, dimension, embeddings_path,
                                             lambda_seq = NULL, verbose = FALSE) {

  # Extract words with non-NA scores for this dimension
  dim_data <- dictionary[!is.na(dictionary[[dimension]]), c("word", dimension)]

  # Load embedding vectors for these words and unit-normalize them
  word_vectors <- .load_seed_vectors(
    seed_words = dim_data$word,
    embeddings_path = embeddings_path,
    verbose = FALSE
  )
  word_vectors <- lapply(word_vectors, .unit_norm)

  # Some dictionary words may not exist in embeddings — filter to matches only
  found_words <- names(word_vectors)
  dim_data <- dim_data[tolower(dim_data$word) %in% found_words, ]

  if (nrow(dim_data) == 0) {
    stop("No dictionary words found in embeddings for dimension: ", dimension)
  }

  # Assemble the W matrix (rows = words, cols = embedding dims) and y vector
  # (target scores) in the same word order for GCV computation
  W <- do.call(rbind, word_vectors[tolower(dim_data$word)])
  y <- as.numeric(dim_data[[dimension]])

  # Delegate to the core SVD-based GCV function
  result <- .gcv_select_lambda(W, y, lambda_seq = lambda_seq, return_all = TRUE)

  if (verbose) {
    cli::cli_alert_info(
      "Dimension '{dimension}': optimal lambda = {signif(result$optimal_lambda, 3)} (df = {round(result$df_values[result$best_idx], 1)})"
    )
  }

  return(result)
}


#' Select optimal lambda via glmnet cross-validation
#'
#' @description
#' Uses glmnet::cv.glmnet to select optimal lambda for elastic net or lasso.
#' This is the equivalent of GCV but for penalized regression methods where
#' no closed-form LOO formula exists.
#'
#' @param dictionary Data frame with 'word' column and dimension columns
#' @param embeddings_path Path to embeddings file
#' @param dimensions Character vector of dimension names
#' @param method "elastic_net" or "lasso"
#' @param l1_ratio Elastic net mixing parameter
#' @param verbose Print progress
#'
#' @return Optimal lambda value
#'
#' @keywords internal
.cv_select_lambda_glmnet <- function(dictionary, embeddings_path, dimensions,
                                     method = "elastic_net", l1_ratio = 0.5,
                                     verbose = TRUE) {

  # Map our API naming to glmnet's: our l1_ratio → glmnet's alpha (mixing param)
  glmnet_alpha <- if (method == "lasso") 1 else l1_ratio

  # One optimal lambda per dimension, then aggregated via median
  optimal_lambdas <- numeric(length(dimensions))
  names(optimal_lambdas) <- dimensions

  for (i in seq_along(dimensions)) {
    dim <- dimensions[i]

    if (verbose) {
      cli::cli_progress_step("Processing dimension: {dim}", spinner = TRUE)
    }

    dim_data <- dictionary[!is.na(dictionary[[dim]]), c("word", dim)]
    word_vectors <- .load_seed_vectors(
      seed_words = dim_data$word,
      embeddings_path = embeddings_path,
      verbose = FALSE
    )

    word_vectors <- lapply(word_vectors, .unit_norm)

    found_words <- names(word_vectors)
    dim_data <- dim_data[tolower(dim_data$word) %in% found_words, ]

    if (nrow(dim_data) < 3) {
      if (verbose) cli::cli_alert_warning("Too few words for CV on dimension '{dim}', using lambda = 0.1")
      optimal_lambdas[i] <- 0.1
      if (verbose) cli::cli_progress_done()
      next
    }

    W <- do.call(rbind, word_vectors[tolower(dim_data$word)])
    y <- as.numeric(dim_data[[dim]])

    # Adapt k-fold CV to dictionary size: small dictionaries may have fewer
    # words than the default 10 folds, so we cap at nrow(W). Minimum 3 folds
    # to get a meaningful CV estimate.
    nfolds <- min(nrow(W), 10)
    if (nfolds < 3) nfolds <- 3

    cv_result <- tryCatch(
      glmnet::cv.glmnet(
        x = W, y = y,
        alpha = glmnet_alpha,
        nfolds = nfolds,
        standardize = FALSE,
        intercept = FALSE
      ),
      error = function(e) NULL
    )

    if (is.null(cv_result)) {
      if (verbose) cli::cli_alert_warning("CV failed for dimension '{dim}', using lambda = 0.1")
      optimal_lambdas[i] <- 0.1
    } else {
      # Use lambda.1se (most regularized lambda within 1 SE of minimum CV error)
      # rather than lambda.min, for a more robust (less overfit) axis
      optimal_lambdas[i] <- cv_result$lambda.1se
    }

    if (verbose) {
      cli::cli_progress_done()
      cli::cli_alert_info("  {dim}: lambda = {signif(optimal_lambdas[i], 3)}")
    }
  }

  # Aggregate: use median across dimensions
  lambda <- median(optimal_lambdas)

  if (verbose) {
    cli::cli_alert_success("Selected lambda = {signif(lambda, 3)} (median across {length(dimensions)} dimension{?s})")
  }

  return(lambda)
}


#' Validate lambda parameter
#'
#' @description
#' Evaluates a vectionary built with a specific lambda by checking:
#' 1. Validity: R² between normalized dictionary scores and projections
#' 2. Differentiation: Average correlation between axes (lower = better)
#'
#' @param vect A built vectionary object
#' @param dictionary Original dictionary data frame
#' @param dimensions Dimension names
#' @param embeddings_path Path to embeddings file (needed for AUC with binary dicts)
#' @param seed Integer seed for reproducibility (passed to random vector sampling)
#'
#' @return List with validity (R² or AUC), differentiation metrics
#'
#' @keywords internal
.validate_lambda <- function(vect, dictionary, dimensions, embeddings_path = NULL, seed) {

  # ── Determine validation metric ──
  # Binary dictionaries (scores are all 0 or 1): use AUC via Mann-Whitney U
  #   - Measures separation between dictionary words and random background words
  #   - 0.5 = random, 1.0 = perfect separation
  # Continuous dictionaries (graded scores): use R² (correlation squared)
  #   - Measures agreement between min-max normalized scores and projections
  first_dim <- dimensions[1]
  dict_scores_check <- dictionary[[first_dim]][!is.na(dictionary[[first_dim]])]
  is_binary <- all(dict_scores_check %in% c(0, 1))

  # For binary dictionaries, sample random words once for AUC computation.
  # These random words serve as the "negative class" (non-dictionary words)
  # for the Mann-Whitney U test.
  random_vectors <- NULL
  if (is_binary && !is.null(embeddings_path)) {
    random_vectors <- .sample_random_vectors(
      embeddings_path = embeddings_path,
      n_sample = 500,
      exclude_words = dictionary$word,
      seed = seed
    )
  }

  # Compute validity independently per dimension
  validities <- sapply(dimensions, function(dim) {
    # Find words present in both the original dictionary and the projections
    dict_words <- dictionary$word[!is.na(dictionary[[dim]])]
    proj_words <- vect$word_projections$word
    common_words <- intersect(tolower(dict_words), tolower(proj_words))

    if (length(common_words) == 0) return(0)

    # Get dictionary scores for common words
    dict_scores <- dictionary[match(common_words, tolower(dictionary$word)), dim]

    # Check if this dimension is binary (all scores 0 or 1) or continuous
    dim_is_binary <- all(dict_scores %in% c(0, 1))

    if (!dim_is_binary) {
      # Continuous dictionary: R² between min-max normalized scores and projections.
      # Min-max normalization ensures both are on [0,1] before comparing.
      proj_scores <- vect$word_projections[match(common_words, tolower(vect$word_projections$word)), dim]

      # If all scores are identical (constant), correlation is undefined → return NA
      if (length(unique(dict_scores)) == 1 || length(unique(proj_scores)) == 1) {
        cli_alert_warning("Dimension '{dim}': constant scores, R² undefined (returning NA)")
        return(NA_real_)
      }

      # Min-max normalization to [0, 1]
      normalize <- function(x) {
        (x - min(x)) / (max(x) - min(x))
      }

      dict_norm <- normalize(dict_scores)
      proj_norm <- normalize(proj_scores)

      # R-squared: correlation^2
      validity <- cor(dict_norm, proj_norm)^2

    } else {
      # Binary/single-class dictionary: use AUC
      if (is.null(random_vectors) || length(random_vectors) == 0) {
        return(0.5)
      }

      # Get projections for dictionary words
      dict_proj <- vect$word_projections[match(common_words, tolower(vect$word_projections$word)), dim]

      # Project random words onto axis
      axis <- vect$axes[[dim]]
      random_proj <- sapply(random_vectors, function(v) sum(v * axis))

      # AUC via Mann-Whitney U statistic:
      # For each (dict_word, random_word) pair, count how often the dictionary
      # word projects higher. Ties count as 0.5. AUC = U / (n_dict * n_random).
      n_dict <- length(dict_proj)
      n_random <- length(random_proj)

      u_stat <- sum(sapply(dict_proj, function(d) sum(d > random_proj) + 0.5 * sum(d == random_proj)))

      validity <- u_stat / (n_dict * n_random)
    }

    return(validity)
  })

  # Determine which metric was used (check first dimension)
  # Consistent with the is_binary check at the top of this function
  validity_metric <- if (is_binary) "auc" else "r2"

  # Overall validity = worst dimension (conservative). NA values from constant
  # scores are ignored so they don't veto otherwise valid lambdas.
  validity <- min(validities, na.rm = TRUE)

  # Axis differentiation: average absolute correlation between all pairs of axes.
  # Low correlation means axes capture distinct semantic dimensions (good).
  # High correlation means axes are redundant (bad). Only meaningful with 2+ dims.
  if (length(vect$axes) >= 2) {
    axes_matrix <- do.call(cbind, vect$axes)
    correlations <- cor(axes_matrix)
    avg_correlation <- mean(abs(correlations[upper.tri(correlations)]))
    differentiation <- 1 - avg_correlation
  } else {
    avg_correlation <- NA_real_
    differentiation <- NA_real_
  }

  return(list(
    validity = validity,
    validity_metric = validity_metric,
    dimension_validities = validities,
    differentiation = differentiation,
    avg_correlation = avg_correlation
  ))
}

#' Select optimal lambda automatically
#'
#' @description
#' Tests multiple lambda values and selects the one with:
#' 1. Validity >= min_validity (normalized score agreement within tolerance)
#' 2. Lowest axis correlation (best differentiation)
#'
#' @param dictionary Data frame with 'word' column and dimension columns
#' @param embeddings_path Path to embeddings file
#' @param dimensions Character vector of dimension names
#' @param method Optimization method
#' @param l1_ratio Elastic net mixing parameter (0 = ridge, 1 = lasso)
#' @param lambda_range Vector of lambda values to test
#' @param min_validity Minimum validity required (0-1, default: 0.75)
#' @param verbose Print progress
#' @param seed Integer seed for reproducibility
#'
#' @return Optimal lambda value
#'
#' @keywords internal
.select_optimal_lambda <- function(
    dictionary,
    embeddings_path,
    dimensions,
    method = "ridge",
    l1_ratio = 0.5,
    lambda_range = c(0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1),
    min_validity = 0.75,
    verbose = TRUE,
    seed
) {

  # Pre-check: lambda = 0 means OLS (no regularization). This requires
  # n_words >= embedding_dims for the system to be fully determined.
  # If not, remove lambda = 0 from the grid to avoid singularity.
  if (0 %in% lambda_range) {
    sample_vectors <- .load_seed_vectors(
      seed_words = dictionary$word[1],
      embeddings_path = embeddings_path,
      verbose = FALSE
    )
    if (length(sample_vectors) > 0) {
      embed_dims <- length(sample_vectors[[1]])
      n_words <- nrow(dictionary)
      if (n_words < embed_dims) {
        lambda_range <- lambda_range[lambda_range != 0]
        if (verbose) {
          cli_alert_info("Skipping lambda = 0 (OLS): dictionary has {n_words} words but embeddings have {embed_dims} dimensions. Use lambda = 0 to force OLS with pseudo-inverse.")
        }
      }
    }
  }

  # Detect if using AUC (binary dict) or R² (continuous dict)
  # Binary = all non-NA scores are 0 or 1 (covers both word-list dicts with all 1s

  # and data-frame dicts with explicit 0s and 1s after binary_word conversion)
  first_dim <- dimensions[1]
  dict_scores_check <- dictionary[[first_dim]][!is.na(dictionary[[first_dim]])]
  using_auc <- all(dict_scores_check %in% c(0, 1))

  if (verbose) {
    cli_alert_info("Testing lambda values: {paste(lambda_range, collapse = ', ')}")
    if (using_auc) {
      cli_alert_info("Validity metric: AUC (binary dictionary detected)")
      cli_alert_info("Minimum AUC required: {round(min_validity * 100, 1)}% (50% = random)")
    } else {
      cli_alert_info("Validity metric: R² (continuous dictionary)")
      cli_alert_info("Minimum R² required: {round(min_validity * 100, 1)}%")
    }
    cli_text("")
  }

  results <- list()

  for (lambda in lambda_range) {
    if (verbose) {
      cli_progress_step("Testing lambda = {lambda}", spinner = TRUE)
    }

    # Build vectionary with this lambda
    vect_result <- .build_vectionary_internal(
      dictionary = dictionary,
      embeddings_path = embeddings_path,
      dimensions = dimensions,
      method = method,
      lambda = lambda,
      l1_ratio = l1_ratio,
      verbose = FALSE,
      seed = seed
    )

    # Create temporary vectionary object
    vect <- list(
      axes = vect_result$axes,
      word_projections = vect_result$word_projections,
      dimensions = dimensions
    )

    # Validate
    validation <- .validate_lambda(vect, dictionary, dimensions, embeddings_path, seed = seed)

    results[[as.character(lambda)]] <- list(
      lambda = lambda,
      validity = validation$validity,
      validity_metric = validation$validity_metric,
      differentiation = validation$differentiation,
      avg_correlation = validation$avg_correlation
    )

    if (verbose) {
      cli_progress_done()
      metric_label <- if (validation$validity_metric == "auc") "AUC" else "R²"
      if (is.na(validation$differentiation)) {
        cli_alert_info("  {metric_label}: {round(validation$validity * 100, 1)}%")
      } else {
        cli_alert_info("  {metric_label}: {round(validation$validity * 100, 1)}%, Axis Differentiation: {round(validation$differentiation * 100, 1)}%")
      }
    }
  }

  if (verbose) {
    cli_text("")
  }

  # ── Selection logic ──
  # 1. Filter to lambdas meeting the minimum validity threshold
  # 2. Among valid lambdas, pick the one with lowest axis correlation (best
  #    differentiation between dimensions)
  # 3. If no lambda meets the threshold, fall back to best-validity lambda
  valid_lambdas <- Filter(function(r) r$validity >= min_validity, results)

  if (length(valid_lambdas) == 0) {
    if (verbose) {
      cli_alert_warning("No lambda meets {round(min_validity * 100, 1)}% validity threshold")
      cli_alert_info("Selecting lambda with best validity")
    }

    best_idx <- which.max(sapply(results, function(r) r$validity))
    best_result <- results[[best_idx]]

  } else {
    # Select lambda with lowest correlation (best differentiation)
    if (verbose) {
      cli_alert_success("Found {length(valid_lambdas)} valid lambda value{?s}")
    }

    correlations <- sapply(valid_lambdas, function(r) r$avg_correlation)

    # If all correlations are NA (single dimension), pick highest validity
    if (all(is.na(correlations))) {
      best_idx <- which.max(sapply(valid_lambdas, function(r) r$validity))
    } else {
      best_idx <- which.min(correlations)
    }

    best_result <- valid_lambdas[[best_idx]]
  }

  if (verbose) {
    metric_label <- if (best_result$validity_metric == "auc") "AUC" else "R²"
    cli_alert_success("Selected lambda = {best_result$lambda}")
    cli_alert_info("  {metric_label}: {round(best_result$validity * 100, 1)}%")
    if (!is.na(best_result$differentiation)) {
      cli_alert_info("  Axis Differentiation: {round(best_result$differentiation * 100, 1)}%")
    }
    cli_text("")
  }

  return(best_result$lambda)
}


#' Build a vec-tionary internal implementation
#'
#' @description
#' Internal function that performs the actual vec-tionary building.
#'
#' @param dictionary Data frame with 'word' column and dimension columns
#' @param embeddings_path Path to embeddings file
#' @param dimensions Character vector of dimension names
#' @param method Optimization method: "ridge", "elastic_net", "lasso", or "duan"
#' @param lambda Regularization parameter (ignored for duan)
#' @param l1_ratio Elastic net mixing parameter (0 = ridge, 1 = lasso)
#' @param project_full_vocab Logical. If TRUE (default), projects ALL words from
#'   embeddings onto learned axes. If FALSE, only returns axes and dictionary
#'   word projections (faster, for preliminary builds).
#' @param verbose Logical, print progress
#' @param seed Seed for reproducibility (passed to .solve_axis_duan)
#'
#' @return List with axes, word_projections, words_found
#'
#' @keywords internal
.build_vectionary_internal <- function(
    dictionary,
    embeddings_path,
    dimensions,
    method = "ridge",
    lambda,
    l1_ratio = 0.5,
    project_full_vocab = TRUE,
    verbose = TRUE,
    seed
) {

  # ── Step A: Load seed word embeddings ──
  # Extract unique dictionary words and find their vectors in the embeddings file.
  seed_words <- unique(dictionary$word)
  if (verbose) cli_progress_step("Loading word vectors", spinner = TRUE)

  word_vectors <- .load_seed_vectors(
    seed_words = seed_words,
    embeddings_path = embeddings_path,
    verbose = FALSE
  )

  if (verbose) cli_progress_done()

  # ── Step B: Unit-normalize embeddings ──
  # FastText word norms correlate with word frequency — frequent words have
  # inflated norms that dominate raw dot products. Unit-normalizing moves all
  # vectors to the unit sphere, so projections become cosine-similarity-based.
  # The axis is NOT unit-normalized (its magnitude encodes dictionary score scale).
  word_vectors <- lapply(word_vectors, .unit_norm)

  # Not all dictionary words exist in the embeddings (misspellings, rare words)
  found_words <- names(word_vectors)
  dictionary_filtered <- dictionary[tolower(dictionary$word) %in% found_words, ]

  if (nrow(dictionary_filtered) == 0) {
    stop("No dictionary words found in embeddings!")
  }

  # ── Step C: Learn one axis per dimension ──
  # For each dimension, extract the words and scores, then solve the regression
  # problem to find the axis vector in embedding space. The method determines
  # which solver is used (ridge, elastic net, lasso, or Duan).
  if (verbose) cli_progress_step("Computing axes for {length(dimensions)} dimension{?s}", spinner = TRUE)

  axes <- list()

  for (dim in dimensions) {
    dim_data <- dictionary_filtered[, c("word", dim)]
    dim_data$word <- tolower(dim_data$word)
    dim_data <- dim_data[!is.na(dim_data[[dim]]), ]

    if (nrow(dim_data) == 0) {
      warning("No valid data for dimension: ", dim)
      next
    }

    word_list <- dim_data$word
    scores <- dim_data[[dim]]
    vectors <- word_vectors[word_list]

    axis <- switch(method,
      "ridge" = .solve_axis_ridge(
        word_vectors = vectors,
        word_scores = scores,
        lambda = lambda
      ),
      "elastic_net" = .solve_axis_elastic_net(
        word_vectors = vectors,
        word_scores = scores,
        lambda = lambda,
        l1_ratio = l1_ratio
      ),
      "lasso" = .solve_axis_elastic_net(
        word_vectors = vectors,
        word_scores = scores,
        lambda = lambda,
        l1_ratio = 1.0
      ),
      "duan" = .solve_axis_duan(
        word_vectors = vectors,
        word_scores = scores,
        dim_name = dim,
        seed = seed
      ),
      stop("Unknown method: ", method, ". Must be 'ridge', 'elastic_net', 'lasso', or 'duan'")
    )

    axes[[dim]] <- axis
  }

  if (verbose) cli_progress_done()

  # ── Step D: Project vocabulary onto learned axes ──
  # project_full_vocab=TRUE: project ALL words from the (filtered) embeddings.
  #   This is the final build step — produces the word_projections data frame
  #   that vectionary_analyze() uses for text scoring.
  # project_full_vocab=FALSE: only project the dictionary words (fast, used for
  #   preliminary builds during lambda selection where full projection is wasteful).
  if (project_full_vocab) {
    if (verbose) cli_progress_step("Projecting full vocabulary", spinner = TRUE)

    word_projections_df <- .project_all_embeddings(
      axes = axes,
      dimensions = dimensions,
      embeddings_path = embeddings_path,
      verbose = FALSE
    )

    if (verbose) {
      cli_progress_done()
      cli_alert_success("Projected {nrow(word_projections_df)} words onto {length(dimensions)} dimension{?s}")
    }
  } else {
    # Only project dictionary words (faster, for preliminary builds)
    word_projections_df <- data.frame(
      word = names(word_vectors),
      stringsAsFactors = FALSE
    )
    for (dim in dimensions) {
      word_projections_df[[dim]] <- sapply(word_vectors, function(v) sum(v * axes[[dim]]))
    }
  }

  # Return results
  result <- list(
    axes = axes,
    word_projections = word_projections_df,
    words_found = length(found_words)
  )

  return(result)
}

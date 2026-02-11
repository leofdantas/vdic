#' vdic: Build and Use Vec-tionaries for Text Analysis
#'
#' @description
#' A framework for building and using vector-based dictionaries (vec-tionaries)
#' for text analysis. Provide seed words scored on dimensions of interest
#' (e.g., moral foundations, sentiment, populism), and vdic learns semantic axes
#' in word-embedding space that can score \emph{any} word -- including words
#' absent from the original dictionary.
#'
#' @details
#'
#' \strong{Design philosophy.}
#' The package separates code from data.
#' It ships R code only.
#' Users download word embeddings once (~1--7 GB), build lightweight
#' vec-tionaries (~3 MB), and share them freely without the original
#' embeddings.
#'
#' @section Core Workflow:
#'
#' \strong{Step 1. Download embeddings} (one-time setup):
#' \preformatted{
#' download_embeddings(language = "pt", model = "fasttext")
#' }
#'
#' \strong{Step 2. Build a vec-tionary} from a seed dictionary + embeddings:
#' \preformatted{
#' dictionary <- data.frame(
#'   word = c("proteger", "cuidar", "machucar"),
#'   care = c(0.9, 0.8, -0.8)
#' )
#' my_vect <- vectionary_builder(
#'   dictionary = dictionary,
#'   embeddings = "vdic_data/cc.pt.300.vec"
#' )
#' }
#'
#' \strong{Step 3. Analyze text} using the vec-tionary:
#' \preformatted{
#' my_vect$mean("Devemos proteger as pessoas vulneraveis")
#' vectionary_analyze(my_vect, texts, metric = "mean")
#' }
#'
#' \strong{Step 4. Save and share} (no embeddings needed):
#' \preformatted{
#' saveRDS(my_vect, "my_vectionary.rds")  # ~3 MB
#' loaded <- readRDS("my_vectionary.rds")
#' }
#'
#' @section Exported Functions:
#'
#' \describe{
#'   \item{[download_embeddings()]}{Download pre-trained word embeddings
#'     (FastText for 157 languages, word2vec for English/Portuguese, GloVe for
#'     English).}
#'   \item{[vectionary_builder()]}{Build a vec-tionary by learning semantic
#'     axes from a seed dictionary and projecting the full embedding vocabulary
#'     onto those axes. Supports ridge, elastic net, LASSO, and Duan et al.
#'     (2025) methods.}
#'   \item{[vectionary_analyze()]}{Score one or more documents against a
#'     vec-tionary. Returns per-dimension scores using a chosen aggregation
#'     metric (mean, RMS, SD, SE, top-10, top-20, or all). Optionally
#'     classifies documents via a one-tailed t-test.}
#'   \item{[vectionary_diagnose()]}{Print a diagnostic report showing the
#'     top-scoring words per dimension and whether seed words rank near the
#'     top. Useful for verifying axis quality.}
#' }
#'
#' @section Dollar-Sign Methods:
#'
#' Vec-tionary objects support the following via the \code{$} operator:
#'
#' \describe{
#'   \item{\code{$mean(text)}}{Arithmetic mean of word projections.}
#'   \item{\code{$rms(text)}}{Root mean square (emphasizes high-magnitude
#'     projections).}
#'   \item{\code{$sd(text)}}{Standard deviation of projections within a
#'     document.}
#'   \item{\code{$se(text)}}{Standard error of the mean.}
#'   \item{\code{$top_10(text)}}{Mean of the 10 highest projections.}
#'   \item{\code{$top_20(text)}}{Mean of the 20 highest projections.}
#'   \item{\code{$metrics(text)}}{All six metrics at once.}
#'   \item{\code{$diagnose(n, dimension)}}{Diagnostic report (see
#'     [vectionary_diagnose()]).}
#' }
#'
#' All methods accept a single string or a character vector and return
#' a named list (one element per dimension). Use \code{as.data.frame()}
#' to convert to a data frame.
#'
#' @section Mathematical Background:
#'
#' \strong{Axis learning.}
#' Given seed-word embeddings \eqn{W \in \mathbb{R}^{n \times d}} (unit-normalized)
#' and dictionary scores \eqn{y \in \mathbb{R}^n}, the package learns
#' a semantic axis \eqn{m \in \mathbb{R}^d}:
#'
#' \itemize{
#'   \item \strong{Ridge} (default):
#'     \eqn{m = (W^\top W + \lambda I)^{-1} W^\top y}, where \eqn{\lambda}
#'     is selected automatically via GCV (Golub et al., 1979).
#'   \item \strong{Elastic net / LASSO}: Solved via \code{glmnet::glmnet()}
#'     with \eqn{\lambda} chosen by cross-validation.
#'   \item \strong{Duan et al. (2025)}:
#'     \eqn{\min_m \sum_i (w_i^\top m - s_i)^2} subject to
#'     \eqn{\|m\| = 1}. No regularization; unit-norm constraint.
#' }
#'
#' \strong{Projection.}
#' All word embeddings are normalized to unit Euclidean norm before axis
#' learning and projection.
#' A word's score is the dot product \eqn{\hat{w}_j^\top m}, which equals
#' \eqn{\|m\| \cos\theta_j} -- cosine similarity scaled by the axis
#' magnitude.
#'
#' \strong{Topic classification.}
#' When \code{alpha} is passed to [vectionary_analyze()], documents are
#' classified via a one-tailed t-test.
#' A document is flagged when its score exceeds
#' \eqn{\bar{x} + t_{1-\alpha,\, n-1} \cdot s}, where \eqn{\bar{x}} and
#' \eqn{s} are the corpus mean and sample standard deviation.
#'
#' @section Build Pipeline:
#'
#' [vectionary_builder()] runs a 7-step pipeline:
#' \enumerate{
#'   \item \strong{Filter embedding vocabulary} -- Remove non-alphabetic tokens,
#'     stopwords, and misspelled words (hunspell).
#'   \item \strong{Expand stem patterns} -- Match patterns like \code{"abandon*"}
#'     to all inflected forms in the filtered embeddings.
#'   \item \strong{Select lambda} -- Choose the regularization parameter via
#'     GCV (ridge), cross-validation (elastic net / LASSO), or grid search.
#'   \item \strong{Learn axes} -- Solve a regression problem per dimension,
#'     mapping seed-word embeddings to dictionary scores.
#'   \item \strong{Expand vocabulary} (optional) -- Find top-N words with
#'     highest projections and add them to the dictionary.
#'   \item \strong{Rebuild} (if expanded) -- Re-learn axes with the enlarged
#'     dictionary.
#'   \item \strong{Project full vocabulary} -- Dot-product every word in the
#'     filtered embeddings with the learned axes. Package into a
#'     \code{Vec-tionary} object and save.
#' }
#'
#' @section Dependencies:
#'
#' \strong{Required}: \code{R.utils}, \code{cli}, \code{MASS}, \code{glmnet},
#' \code{data.table}.
#'
#' \strong{Optional}:
#' \itemize{
#'   \item \code{hunspell}: Required for \code{spellcheck = TRUE} (default)
#'     in [vectionary_builder()]. Filters embeddings to exclude non-words,
#'     symbols, and web artifacts.
#'   \item \code{alabama}: Required for \code{method = "duan"} (constrained
#'     optimization).
#' }
#'
#' @references
#' Duan, Z., Shao, A., Hu, Y., Lee, H., Liao, X., Suh, Y. J., Kim, J.,
#' et al. (2025). Constructing Vec-Tionaries to Extract Message Features
#' from Texts: A Case Study of Moral Content. \emph{Political Analysis},
#' 1--21. \doi{10.1017/pan.2025.6}.
#'
#' Hopp, F. R., Fisher, J. T., Cornell, D., Huskey, R., & Weber, R. (2021).
#' The extended Moral Foundations Dictionary (eMFD): Development and
#' applications of a crowd-sourced approach to extracting moral intuitions
#' from text. \emph{Behavior Research Methods}, 53(1), 232--246.
#'
#' @seealso
#' \itemize{
#'   \item [download_embeddings()] for obtaining word embeddings
#'   \item [vectionary_builder()] for building custom vec-tionaries
#'   \item [vectionary_analyze()] for scoring text
#'   \item [vectionary_diagnose()] for diagnostic reports
#' }
#'
#' @author Leonardo Dantas
#'
#' @import cli
#' @keywords internal
"_PACKAGE"

# Package-level variables
.vdic_env <- new.env(parent = emptyenv())

#- Stopword Lists ----
# Common stopwords for supported languages
# These are removed by default during text analysis to improve signal

#' @keywords internal
.stopwords <- list(
  en = c(
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "where", "why", "how", "all",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "can", "will", "should", "now", "i", "me", "my", "myself", "we",
    "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "would",
    "could", "might", "must", "shall", "of", "as"
  ),
  pt = c(
    "a", "o", "e", "de", "da", "do", "das", "dos", "em", "na", "no",
    "nas", "nos", "um", "uma", "uns", "umas", "para", "por", "com",
    "sem", "sob", "sobre", "entre", "até", "após", "ante", "contra",
    "desde", "perante", "que", "se", "não", "mais", "mas", "como",
    "ou", "ao", "aos", "às", "pela", "pelo", "pelas", "pelos", "esse",
    "essa", "esses", "essas", "este", "esta", "estes", "estas", "isso",
    "isto", "aquele", "aquela", "aqueles", "aquelas", "aquilo", "ele",
    "ela", "eles", "elas", "eu", "tu", "você", "vocês", "nós", "vós",
    "me", "te", "lhe", "nos", "vos", "lhes", "meu", "minha", "meus",
    "minhas", "teu", "tua", "teus", "tuas", "seu", "sua", "seus", "suas",
    "nosso", "nossa", "nossos", "nossas", "vosso", "vossa", "vossos",
    "vossas", "qual", "quais", "quanto", "quanta", "quantos", "quantas",
    "quem", "onde", "quando", "porque", "porquê", "já", "ainda", "também",
    "só", "bem", "muito", "pouco", "mais", "menos", "tão", "tanto",
    "tanta", "tantos", "tantas", "ser", "estar", "ter", "haver", "ir",
    "vir", "fazer", "dizer", "ver", "dar", "saber", "querer", "poder",
    "dever", "foi", "foram", "era", "eram", "será", "serão", "seria",
    "seriam", "está", "estão", "estava", "estavam", "tem", "têm", "tinha",
    "tinham", "há", "houve", "havia", "são", "é"
  ),
  es = c(
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como",
    "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
    "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era",
    "esa", "esas", "ese", "eso", "esos", "esta", "estado", "estas",
    "este", "esto", "estos", "fue", "fueron", "ha", "había", "han",
    "hasta", "hay", "la", "las", "le", "les", "lo", "los", "más", "me",
    "mi", "mis", "muy", "nada", "ni", "no", "nos", "nosotros", "nuestra",
    "nuestras", "nuestro", "nuestros", "o", "os", "otra", "otras", "otro",
    "otros", "para", "pero", "poco", "por", "porque", "que", "quien",
    "se", "sea", "según", "ser", "si", "sido", "sin", "sobre", "son",
    "su", "sus", "también", "tan", "tanto", "te", "ti", "tiene", "toda",
    "todas", "todo", "todos", "tu", "tus", "un", "una", "uno", "unos",
    "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros",
    "y", "ya", "yo"
  )
)

#' Get stopwords for a language
#'
#' @param lang Language code (e.g., "en", "pt", "es")
#' @return Character vector of stopwords, or NULL if language not supported
#' @keywords internal
.get_stopwords <- function(lang) {
  if (lang %in% names(.stopwords)) {
    return(.stopwords[[lang]])
  }
  return(NULL)
}


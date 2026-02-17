source("tests/vdic-builder.R")
library(vdic)
library(reticulate)
reticulate::py_config()

vdic::download_embeddings(model = "siglip")

env_dict <- c("mudança", "mudanças", "clima","climático", "climatológico","aquecimento", "globo","global", "crise", "ambiente", "carbono", "temperatura", "atmosfera", "emissões", "gases", "sustentabilidade", "renovável", "fósseis", "descarbonização", "estufa",  "ambiental", "antropogênico", "antropogênica")
lgbt_dict <- c(                                                  
    # Orientations and gender identities
    "lgbt", "lgbtq", "gay", "lesbian", "bisexual", "transgender", "queer",          
    "intersex", "asexual", "pansexual", "nonbinary", "genderqueer",
    "genderfluid", "agender", "heterosexual", "homosexual", "demisexual", "aromantic",
    "graysexual", "omnisexual", "polysexual", "androgynous",
    "demiromantic", "cisgender", "trans", "heteroflexible",
    "homoflexible", "rainbow")
vect <- vdic::vectionary_builder(
  dictionary = lgbt_dict,
  embeddings = "siglip",
  modality = "multimodal",
  language = "en",
  binary_word = TRUE,
  method = "ridge",
  spellcheck = TRUE,
  remove_stopwords = TRUE,
  save_path = "lgbt_vect.rds",
  seed = 574
)
lgbt_dict2 <- c(                                                  
    # Orientations and gender identities
    "lgbt")
vect <- vdic::vectionary_builder(
  dictionary = lgbt_dict2,
  embeddings = "siglip",
  modality = "multimodal",
  language = "en",
  binary_word = TRUE,
  method = "ridge",
  spellcheck = TRUE,
  remove_stopwords = TRUE,
  save_path = "vdic_data/lgbt_vect2.rds",
  seed = 574
)

images <- list.files("/Users/leodantas/RPackages/vdic/tests/images", full.names = TRUE)

res <- analyze_image(
  vect = vect,
  images = images, 
  alpha = 0.25
  )

res$image <- gsub("/Users/leodantas/RPackages/vdic/tests/images/", "", res$image)
res$image <- gsub("\\.jpg|\\.webp|\\.png", "", res$image)
res

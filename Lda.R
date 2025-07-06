#=== 1. Install and Load Required Libraries ===
install.packages(c("tm", "textstem", "text2vec", "topicmodels", 
                   "betareg", "randomForest", "caret", "dplyr", "readxl", "Matrix"))
library(readxl)
library(tm)
library(textstem)
library(text2vec)
library(topicmodels)
library(betareg)
library(randomForest)
library(caret)
library(dplyr)
library(Matrix)

#=== 2. Load Data ===
narratives <- read.csv("narratives (1).csv", stringsAsFactors=FALSE)
metadata   <- read_excel("metadata.xlsx")

#=== 3. Text Preprocessing ===
corpus <- VCorpus(VectorSource(narratives$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, content_transformer(lemmatize_strings))
clean_text <- sapply(corpus, as.character)

#=== 4. Create Document-Term Matrix (DTM) ===
prep_fun <- tolower
tok_fun  <- word_tokenizer
it <- itoken(clean_text, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 5)
vectorizer <- vocab_vectorizer(vocab)
dtm <- create_dtm(it, vectorizer)
rownames(dtm) <- narratives$doc_id
dtm <- dtm[rowSums(dtm) > 0, ]

#=== 5. Fit LDA Model ===
K <- 5
lda_model <- LDA(dtm, k = K, control = list(seed = 1234))
lda_topics <- posterior(lda_model)$topics

#=== 6. Topic Proportions dataframe ===
topic_props <- as.data.frame(lda_topics)
topic_props$doc_id <- rownames(dtm)
colnames(topic_props)[1:K] <- paste0("Topic", 1:K)
write.csv(topic_props, "lda_topic_proportions.csv", row.names = FALSE)
# Number of top words you want
top_n <- 10  

# Extract top words per topic
lda_top_terms <- terms(lda_model, top_n)

# Print nicely
for (i in 1:K) {
  cat(paste0("\n==== Topic ", i, " ====\n"))
  cat(paste(lda_top_terms[, i], collapse = ", "))
  cat("\n")
}

#=== 7. Merge with Metadata ===
combined_data <- merge(metadata, topic_props, by.x="Crash.Instance", by.y="doc_id", all.x=TRUE)

#=== 8. Handle Missing Topic Proportions (set NAs to 0) ===
topic_cols <- paste0("Topic", 1:K)
combined_data[topic_cols] <- lapply(combined_data[topic_cols], function(x) ifelse(is.na(x), 0, x))

#=== 9. Prepare variables for Beta Regression ===
model_vars <- c("Highway.Class", "Speed.Limit.at.Crash.Site", "Relation.to.Roadway",
                "Road.Conditions", "Weather.Conditions..2004.2015.",
                "Lighting.Conditions", "Crash.Type", "Total.Motor.Vehicles",
                "Crash.Month", "Crash.Day")

# Keep only complete cases
combined_data_beta <- combined_data[complete.cases(combined_data[, model_vars]), ]

# Convert categorical vars to factor
for (v in c("Highway.Class", "Relation.to.Roadway", "Road.Conditions",
            "Weather.Conditions..2004.2015.", "Lighting.Conditions",
            "Crash.Type", "Crash.Month", "Crash.Day")) {
  combined_data_beta[[v]] <- as.factor(combined_data_beta[[v]])
}

# Adjust topic proportions to avoid 0/1
epsilon <- 1e-4
combined_data_beta[topic_cols] <- lapply(combined_data_beta[topic_cols],
                                         function(x) pmin(pmax(x, epsilon), 1 - epsilon))

#=== 10. Beta Regression ===
for (t in 1:K) {
  topic_col <- paste0("Topic", t)
  formula_str <- paste(topic_col, "~", paste(model_vars, collapse = " + "))
  beta_model <- betareg(as.formula(formula_str), data=combined_data_beta)
  print(summary(beta_model))
  png(filename = paste0("BetaDiag_", topic_col, ".png"))
  plot(beta_model)
  dev.off()
}

#=== 11. Save Merged Data ===
write.csv(combined_data, "lda_combined_metadata_topicpredictions.csv", row.names=FALSE)
# Split your DTM into train/test sets
set.seed(123)
train_ratio <- 0.8
train_indices <- sample(1:nrow(dtm), size = floor(train_ratio * nrow(dtm)))
dtm_train <- dtm[train_indices, ]
dtm_test  <- dtm[-train_indices, ]

# Fit LDA model on training data
K <- 5
lda_model_train <- LDA(dtm_train, k = K, control = list(seed = 1234))

# Evaluate held-out log-likelihood on test data
heldout_loglik <- logLik(LDA(dtm_test, model = lda_model_train))
cat("Held-out log-likelihood (LDA):", heldout_loglik, "\n")

total_heldout_words <- sum(dtm_test)
per_word_loglik <- as.numeric(heldout_loglik) / total_heldout_words
cat("Per-word log-likelihood (LDA):", per_word_loglik, "\n")
# Coherence
library(text2vec)

# Create iterators for your cleaned text
it <- itoken(clean_text, tokenizer = word_tokenizer, progressbar = FALSE)

# Build Term Co-occurrence Matrix (TCM)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 5)
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5)  # window size 5 or adjust

# Now compute topic coherence for each topic
lda_terms <- terms(lda_model, 10)  # top 10 terms per topic
K <- 5
coherence_scores <- sapply(1:K, function(k) {
  topic_words <- lda_terms[, k]
  coherence(tcm, topic_words)
})

# Print results
print(coherence_scores)
cat("Mean Coherence (LDA):", mean(coherence_scores), "\n")

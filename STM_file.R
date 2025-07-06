#=== 1. Install and Load Required Libraries ===
install.packages(c("stm", "tm", "textstem", "readr", "readxl", "dplyr"))
library(stm)
library(tm)
library(textstem)
library(readr)
library(readxl)
library(dplyr)

#=== 2. Load Data ===
narratives <- read_csv("narratives (1).csv")
metadata <- read_excel("metadata.xlsx")
metadata <- as.data.frame(metadata)

# Merge on doc_id = Crash.Instance
combined_data <- merge(narratives, metadata, by.x = "doc_id", by.y = "Crash.Instance")

#=== 3. Text Preprocessing ===
corpus <- VCorpus(VectorSource(combined_data$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, content_transformer(lemmatize_strings))
clean_text <- sapply(corpus, as.character)

#=== 4. Prepare Documents for STM ===
processed <- textProcessor(documents = clean_text, metadata = combined_data, verbose = TRUE)
prep <- prepDocuments(processed$documents, processed$vocab, processed$meta, verbose = TRUE)

docs <- prep$documents
vocab <- prep$vocab
meta <- prep$meta

#=== 5. Convert Categorical Variables to Factor ===
covariates <- c("Highway.Class", "Speed.Limit.at.Crash.Site", "Relation.to.Roadway",
                "Road.Conditions", "Weather.Conditions..2004.2015.", "Lighting.Conditions",
                "Crash.Type", "Total.Motor.Vehicles", "Crash.Month", "Crash.Day")

for (var in covariates) {
  if (!is.numeric(meta[[var]])) {
    meta[[var]] <- as.factor(meta[[var]])
  }
}

#=== 6. Fit STM Model ===
K <- 5
formula_str <- as.formula(paste("~", paste(covariates, collapse = " + ")))

stm_model <- stm(documents = docs, vocab = vocab, 
                 K = K, 
                 prevalence = formula_str, 
                 data = meta, seed = 1234)

#=== 7. Show Top Words per Topic ===
labelTopics(stm_model, n = 10)

#=== 8. Estimate Metadata Effects on Topic Prevalence ===
effects_model <- estimateEffect(1:K ~ ., stm_model, meta = meta[, c(covariates, "doc_id")], uncertainty = "Global")
summary(effects_model)

#=== 9. Save Topic Proportions ===
topic_proportions <- stm_model$theta
topic_proportions_df <- as.data.frame(topic_proportions)
topic_proportions_df$doc_id <- meta$doc_id
write.csv(topic_proportions_df, "stm_topic_proportions.csv", row.names = FALSE)

#=== 10. Save STM Model Object ===
saveRDS(stm_model, "final_stm_model.rds")
# Saving p values
saveRDS(effects_model, "stm_metadata_effects_model.rds")
#Ploting 
# List of covariates you want to plot effects for
covariates_to_plot <- c("Highway.Class", "Lighting.Conditions", "Crash.Type")

# Create effect plots for each topic and each covariate
for (k in 1:K) {
  for (covar in covariates_to_plot) {
    png(filename = paste0("STM_Effect_Topic", k, "_", gsub("[^[:alnum:]]", "_", covar), ".png"),
        width = 1000, height = 600)
    
    plot(effects_model, covariate = covar, topics = k, 
         method = "pointestimate", 
         main = paste("Effect of", covar, "on Topic", k),
         xlab = covar)
    
    dev.off()
  }
}

cat("✅ All effect plots saved successfully.\n")
#topics
# Number of top words you want per topic
top_n <- 10

# Extract top words using labelTopics()
top_words <- labelTopics(stm_model, n = top_n)

# View in console
for (k in 1:K) {
  cat(paste0("\n==== Topic ", k, " ====\n"))
  cat(paste(top_words$prob[k, ], collapse = ", "))
  cat("\n")
}

#=== Optional: Save to CSV file ===
# Convert to data frame for saving
top_words_df <- data.frame(Topic = rep(1:K, each = top_n),
                           Word = as.vector(t(top_words$prob)))

# Save as CSV
write.csv(top_words_df, "STM_Top_Words_Per_Topic.csv", row.names = FALSE)

cat("\n✅ Top words for each topic extracted and saved.\n")
# Load your saved effects model
effects_model <- readRDS("stm_metadata_effects_model.rds")

# Get summary of the effects model
eff_summary <- summary(effects_model)

# Extract p-values into a tidy data frame
K <- length(eff_summary$tables)  # Number of topics

pval_list <- lapply(1:K, function(k) {
  pvals <- eff_summary$tables[[k]][, "Pr(>|t|)"]
  data.frame(
    Topic = paste0("Topic_", k),
    Covariate_Level = names(pvals),
    P_Value = pvals,
    row.names = NULL
  )
})

# Combine all topics into one data frame
pval_df <- do.call(rbind, pval_list)

# Save to CSV
write.csv(pval_df, "STM_Topic_Metadata_Pvalues.csv", row.names = FALSE)

cat("✅ P-values extracted from saved model and saved to 'STM_Topic_Metadata_Pvalues.csv'\n")

sc <- semanticCoherence(model=stm_model, documents=docs)
print("Semantic Coherence per topic:")
print(sc)

# Mean coherence score
mean_sc <- mean(sc)
cat("Mean Semantic Coherence across topics:", mean_sc, "\n")

#=== 3. Exclusivity per topic ===
ex <- exclusivity(stm_model)
print("Exclusivity per topic:")
print(ex)

# Mean exclusivity
mean_ex <- mean(ex)
cat("Mean Exclusivity across topics:", mean_ex, "\n")

#=== 4. Held-out Likelihood (model generalizability)
# Create a held-out set split
set.seed(1234)
ho <- make.heldout(documents=docs, vocab=vocab)

# Refit model on training portion
stm_model_ho <- stm(ho$documents, vocab=ho$vocab, K=5, 
                    prevalence=formula_str, data=meta, 
                    seed=1234, max.em.its=50, verbose=FALSE)

# Compute held-out likelihood
heldout <- eval.heldout(stm_model_ho, ho$missing)
cat("Held-out log-likelihood:", heldout$expected.heldout, "\n")
#=== 11. Load topic proportions and metadata ===
topic_proportions_df <- as.data.frame(stm_model$theta)
topic_proportions_df$doc_id <- meta$doc_id

# Merge topic proportions with original metadata
full_data <- merge(meta, topic_proportions_df, by = "doc_id")

# Confirm how many missing values in Crash.Type
table(is.na(full_data$Crash.Type))

#=== 12. Prepare data for Random Forest ===
# Convert factors properly
full_data$Crash.Type <- as.factor(full_data$Crash.Type)
full_data$Highway.Class <- as.factor(full_data$Highway.Class)
full_data$Relation.to.Roadway <- as.factor(full_data$Relation.to.Roadway)

# Select predictors (topics + covariates) and target
predictor_cols <- c(paste0("V", 1:K),  # topic proportions: V1, V2, ..., V5
                    "Highway.Class", "Speed.Limit.at.Crash.Site", "Relation.to.Roadway")

# Subset complete cases for model training
train_data <- full_data[!is.na(full_data$Crash.Type) & complete.cases(full_data[, predictor_cols]), ]

# Subset missing cases for imputation
impute_data <- full_data[is.na(full_data$Crash.Type) & complete.cases(full_data[, predictor_cols]), ]
# Split train_data into train/test for evaluation
set.seed(123)
train_index <- createDataPartition(train_data$Crash.Type, p = 0.8, list = FALSE)
rf_train <- train_data[train_index, ]
rf_test  <- train_data[-train_index, ]

# Train Random Forest classifier
set.seed(1234)
rf_model <- randomForest(as.formula(paste("Crash.Type ~", paste(predictor_cols, collapse = " + "))),
                         data = rf_train, ntree = 500, importance = TRUE)

# Predict on test set
predictions <- predict(rf_model, newdata = rf_test)

# Compute confusion matrix and metrics
cm <- confusionMatrix(predictions, rf_test$Crash.Type)
print(cm)

# Save confusion matrix table to CSV
write.csv(as.data.frame(cm$table), "RF_Confusion_Matrix.csv", row.names=TRUE)

# Save per-class metrics (Sensitivity, Specificity, Precision, F1)
write.csv(as.data.frame(cm$byClass), "RF_Per_Class_Metrics.csv", row.names=TRUE)

# Save overall metrics (Accuracy, Kappa)
write.csv(as.data.frame(t(cm$overall)), "RF_Overall_Metrics.csv", row.names=FALSE)

#=== Impute missing Crash.Type values ===
if (nrow(impute_data) > 0) {
  imputed_values <- predict(rf_model, newdata = impute_data)
  
  # Assign back to original data
  full_data$Crash.Type[match(impute_data$doc_id, full_data$doc_id)] <- as.character(imputed_values)
}

# Confirm no missing values remain
table(is.na(full_data$Crash.Type))

#=== Save final dataset with imputed values ===
write.csv(full_data, "STM_RF_Imputed_Metadata.csv", row.names = FALSE)

cat("✅ Random Forest trained, evaluated, and missing Crash.Type values imputed.\n")

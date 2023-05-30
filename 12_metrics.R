# Load necessary libraries.
library(pacman)
p_load(tidymodels, tidyverse, yardstick, caret, dbarts, 
       xgboost, earth, nnet, kernlab, baguette, brulee, mda, discrim)

# deal with package conflicts
tidymodels_prefer()

# load data
load("data/cleaned.rda")

# load model, preview results, and predict - elastic net
load("results/elastic_cv.rda")
elastic_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
elastic_tuned_workflow <- elastic_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(elastic_tuned, metric = "roc_auc"))

## fit
elastic_results <- fit(elastic_tuned_workflow, class_train)

## predictions
pred_elastic <- predict(elastic_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_elastic, file = "submissions/pred_elastic.csv", row.names = F)

## remove to free up memory
rm(elastic_tuned, elastic_tuned_workflow, elastic_results, pred_elastic)

# load model, preview results, and predict - elastic net, pca
load("results/elastic_pca_cv.rda")
elastic_pca_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
elastic_pca_tuned_workflow <- elastic_pca_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(elastic_pca_tuned, metric = "roc_auc"))

## fit
elastic_pca_results <- fit(elastic_pca_tuned_workflow, class_train)

## predictions
pred_elastic_pca <- predict(elastic_pca_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_elastic_pca, file = "submissions/pred_elastic_pca.csv", row.names = F)

## remove to free up memory
rm(elastic_pca_tuned, elastic_pca_tuned_workflow, 
   elastic_pca_results, pred_elastic_pca)

# load model, preview results, and predict - boosted tree
load("results/bt_cv.rda")
bt_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
bt_tuned_workflow <- bt_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(bt_tuned, metric = "roc_auc"))

## fit
set.seed(22)
bt_results <- fit(bt_tuned_workflow, class_train)

## predictions
pred_bt <- predict(bt_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_bt, file = "submissions/pred_bt.csv", row.names = F)

## remove to free up memory
rm(bt_tuned, bt_tuned_workflow, bt_results, pred_bt)

# load model, preview results, and predict - bart
load("results/bart_cv.rda")
bart_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
bart_tuned_workflow <- bart_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(bart_tuned, metric = "roc_auc"))

## fit
set.seed(142)
bart_results <- fit(bart_tuned_workflow, class_train)

## predictions
pred_bart <- predict(bart_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_bart, file = "submissions/pred_bart.csv", row.names = F)

## remove to free up memory
rm(bart_tuned, bart_tuned_workflow, bart_results, pred_bart)

# load model, preview results, and predict - svm
load("results/svm_rad_cv.rda")
svm_rad_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
svm_rad_tuned_workflow <- svm_rad_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(svm_rad_tuned, metric = "roc_auc"))

## fit
svm_rad_results <- fit(svm_rad_tuned_workflow, class_train)

## predictions
pred_svm_rad <- predict(svm_rad_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_svm_rad, file = "submissions/pred_svm_rad.csv", row.names = F)

## remove to free up memory
rm(svm_rad_tuned, svm_rad_tuned_workflow, svm_rad_results, pred_svm_rad)

# load model, preview results, and predict - mlp sgm
load("results/mlp_sgm_cv.rda")
mlp_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
mlp_tuned_workflow <- mlp_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(mlp_tuned, metric = "roc_auc"))

## fit
set.seed(66)
mlp_results <- fit(mlp_tuned_workflow, class_train)

## predictions
pred_mlp_sgm <- predict(mlp_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_mlp_sgm, file = "submissions/pred_mlp_sgm.csv", row.names = F)

## rename
mlp_sgm_tictoc <- mlp_tictoc

## remove to free up memory
rm(mlp_tuned, mlp_tuned_workflow, mlp_results, pred_mlp_sgm, mlp_tictoc)

## remove to free up memory
rm(svm_rad_tuned, svm_rad_tuned_workflow, svm_rad_results, pred_svm_rad)

# load model, preview results, and predict - mlp bag
load("results/mlp_cv.rda")
mlp_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
mlp_tuned_workflow <- mlp_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(mlp_tuned, metric = "roc_auc"))

## fit
set.seed(71)
mlp_results <- fit(mlp_tuned_workflow, class_train)

## predictions
pred_mlp <- predict(mlp_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_mlp, file = "submissions/pred_mlp.csv", row.names = F)

## remove to free up memory
rm(mlp_tuned, mlp_tuned_workflow, mlp_results, pred_mlp)

# load model, preview results, and predict - mars bag
load("results/mars_cv.rda")
mars_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
mars_tuned_workflow <- mars_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(mars_tuned, metric = "roc_auc"))

## fit
set.seed(63)
mars_results <- fit(mars_tuned_workflow, class_train)

## predictions
pred_mars <- predict(mars_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_mars, file = "submissions/pred_mars.csv", row.names = F)

## remove to free up memory
rm(mars_tuned, mars_tuned_workflow, mars_results, pred_mars)

# load model, preview results, and predict - discriminant analysis
load("results/disc_cv.rda")
disc_tuned %>% 
  show_best(metric = "roc_auc")

## finalize model
disc_tuned_workflow <- disc_tuned %>% extract_workflow() %>% 
  finalize_workflow(select_best(disc_tuned, metric = "roc_auc"))

## fit
disc_results <- fit(disc_tuned_workflow, class_train)

## predictions
pred_disc <- predict(disc_results, new_data = class_test, type = "prob") %>% 
  bind_cols(class_test %>% select(id)) %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_disc, file = "submissions/pred_disc.csv", row.names = F)

## remove to free up memory
rm(disc_tuned, disc_tuned_workflow, disc_results, pred_disc)

# load results and save pred - ensemble
load("results/ensemble_res.rda")

## predictions
pred_ensemble <- pred_ensemble %>% 
  rename(y = .pred_1) %>% 
  select(id, y)

## save
write.csv(pred_ensemble, file = "submissions/pred_ensemble.csv", row.names = F)

# Build tables and plots for final memo
## Table of Model Tuning
models_table <- bind_rows(elastic_tictoc,
                          elastic_pca_tictoc,
                          bt_tictoc,
                          bart_tictoc,
                          svm_rad_tictoc,
                          mlp_sgm_tictoc,
                          mlp_tictoc,
                          mars_tictoc,
                          disc_tictoc) %>% 
  mutate("Run Time (min.)" = round(runtime / 60, 2)) %>% 
  cbind(Recipe = c("Recipe 2", "Recipe 3", "Recipe 1", "Recipe 1", "Recipe 2", "Recipe 1", 
                   "Recipe 1", "Recipe 1", "Recipe 1")) %>% 
  cbind(roc_auc = c(.5907, .6061, .5989, .6112, .5888, # add test performance metrics from Kaggle
                                  .5636, .6061, .6113, .5926)) %>% 
  arrange(desc(roc_auc)) %>% 
  rename(Model = model, "ROC_AUC of Best Tuning" = roc_auc) %>% 
  select(Model, Recipe, "Run Time (min.)", "ROC_AUC of Best Tuning")

## Best Ensemble Table
load("results/ensemble_res.rda") # load in
ens_table <- stack_coef %>% 
  mutate(Penalty = case_when(!is.na(penalty.y) ~ as.character(round(penalty.y, 3)),
                             !is.na(penalty) ~ as.character(round(penalty, 3)),
                             .default = "-"),
         Mixture = case_when(!is.na(mixture.y) ~ as.character(round(mixture.y, 3)),
                             .default = "-"),
         "Hide. Un." = case_when(!is.na(hidden_units) ~ as.character(hidden_units),
                                 .default = "-"),
         Bags = case_when(startsWith(member, "mlp") ~ "50",
                          startsWith(member, "mars") ~ "20",
                          .default = "-"),
         "Num. Terms" = case_when(!is.na(num_terms.x) ~ as.character(num_terms.x),
                                  !is.na(num_terms.y) ~ as.character(num_terms.y),
                                  .default = "-"),
         "Prod. Deg." = case_when(!is.na(prod_degree.x) ~ as.character(prod_degree.x),
                                  !is.na(prod_degree.y) ~ as.character(prod_degree.y),
                                  .default = "-"),
         Trees = case_when(!is.na(trees) ~ as.character(trees),
                                  .default = "-"),
         "Term. Node Coef." = case_when(!is.na(prior_terminal_node_coef) 
                                        ~ as.character(prior_terminal_node_coef),
                             .default = "-"),
         "Term. Node Exp." = case_when(!is.na(prior_terminal_node_expo) 
                                        ~ as.character(prior_terminal_node_expo),
                                        .default = "-"),
         "Stacking Coef." = round(coef, 3)) %>% 
  select(-member) %>% 
  cbind(Member = c("Elastic Net (PCA)", "Elastic Net (PCA)", 
                   "Elastic Net (PCA)", "BART", 
                   "MLP, Bagged", "MARS, Bagged", "MARS, Bagged", 
                   "Disc. Analysis, Flexible")) %>% 
  arrange(desc(coef)) %>% 
  select(Member, Bags, "Num. Terms", "Prod. Deg.", Penalty, Mixture, Trees, 
         "Term. Node Coef.", "Term. Node Exp.", "Hide. Un.",
         "Stacking Coef.") 

## Best Ensemble Plot
ens_plot <- ens_table %>% 
  rename(coef = "Stacking Coef.") %>% 
  cbind(lab = c("h", "g", "f", "e", "d", "c", "b", "a")) %>% 
  ggplot(aes(x = lab, y = coef, fill = Member)) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_minimal() +
  labs(title = "Ensemble Blend",
       y = "Stacking Coefficient",
       x = "Member Model") +
  theme(axis.text.y = element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))

# save
save(models_table, ens_plot, ens_table,
     file = "results/memo_objs.rda")
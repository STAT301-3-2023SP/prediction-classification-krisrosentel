# load libraries
library(pacman)
p_load(tidymodels, tidyverse, yardstick, stacks, dbarts, 
       xgboost, earth, nnet, kernlab, baguette, mda, discrim)

# deal with package conflicts
tidymodels_prefer()

# Load in
load("results/elastic_cv.rda")

# Create stack
class_stack <- stacks() %>%
  add_candidates(elastic_tuned)

# Remove to free up memory 
rm(elastic_tuned, elastic_tictoc)

# Load in
load("results/elastic_pca_cv.rda")

# Add to data stack
class_stack <- class_stack %>%
  add_candidates(elastic_pca_tuned)

# Remove to free up memory 
rm(elastic_pca_tuned, elastic_pca_tictoc)

# Load more
load("results/bart_cv.rda")

# Add to data stack
class_stack <- class_stack %>%
  add_candidates(bart_tuned)

# Remove to free up memory 
rm(bart_tuned, bart_tictoc)

# Load more
load("results/mlp_cv.rda")

# Add to data stack
class_stack <- class_stack %>%
  add_candidates(mlp_tuned)

# Remove to free up memory 
rm(mlp_tuned, mlp_tictoc)

# Load more
load("results/mars_cv.rda")

# Add to data stack
class_stack <- class_stack %>%
  add_candidates(mars_tuned)

# Remove to free up memory 
rm(mars_tuned, mars_tictoc)

# Load more
load("results/disc_cv.rda")

# Add to data stack
class_stack <- class_stack %>%
  add_candidates(disc_tuned)

# Remove to free up memory 
rm(disc_tuned, disc_tictoc)

# view data stack
class_stack # 135 candidate models

# Tune the stack 
## penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

## blend predictions using penalty defined above (tuning step, set seed)
class_blend <-
  class_stack %>%
  blend_predictions(penalty = blend_penalty,
                    metric = metric_set(roc_auc)) # penalty = .000001

# remove to free up memory
rm(class_stack)

# Explore the blended model stacks
autoplot(class_blend)
autoplot(class_blend, type = "weights")

# merge stacking coefs
stack_coef <- collect_parameters(class_blend, "elastic_tuned") %>% 
  full_join(collect_parameters(class_blend, "elastic_pca_tuned"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(class_blend, "bart_tuned"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(class_blend, "mlp_tuned"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(class_blend, "mars_tuned"), 
            by = c("member", "coef")) %>% 
  full_join(collect_parameters(class_blend, "disc_tuned"), 
            by = c("member", "coef")) %>% 
  filter(coef != 0) 

# load data
load("data/cleaned.rda")

# fit ensemble to entire training set 
set.seed(142)
ensemble_results <- class_blend %>%
  fit_members()

# predict
pred_ensemble <- predict(ensemble_results, new_data = class_test, 
                         type = "prob", members = T) %>% 
  bind_cols(class_test %>% select(id)) 
  
# save
save(stack_coef, pred_ensemble,
     file = "results/ensemble_res.rda")
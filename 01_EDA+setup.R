# Load package(s)
library(pacman)
p_load(tidymodels, tidyverse, stringr, skimr, vip, psych)

# handle common conflicts
tidymodels_prefer()

# load data
class_train <- read_csv("data/train.csv") 
class_test <- read_csv("data/test.csv")

# look at balance of outcome. Looks fairly balanced
class_train %>% 
  group_by(y) %>% 
  summarise(count = n()) %>% 
  mutate(prop = count / sum(count))

# look for possible factors
fct_vars <- class_train %>% select_if(function(col) length(unique(col)) < 8) %>% #vars with 7 levels or less
  select_if(function(col) all(col %% 1 == 0)) %>% # vars with only whole nums
  colnames()

# factors vars
class_train <- mutate_at(class_train, c(fct_vars), factor)
class_test <- mutate_at(class_test, c(fct_vars[fct_vars != "y"]), factor)

# save
save(class_train, class_test, file = "data/cleaned.rda")

# run an initial random forest on subset of data to select 40 most important predictors 
## create subset of data - 30% of data
set.seed(51) # set seed
class_split <- initial_split(class_train, prop = 0.3, strata = y)
class_include <- training(class_split) # subset to determine important predictors

##  make list of vars  with any missing
miss_screen <- class_include %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

## recipe 
recipe_screen <- recipe(y ~ ., data = class_include) %>%
  step_rm(id) %>% 
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_impute_knn(miss_screen) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors(), one_hot = T) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  prep()

## set up rf model
rf_screen <- rand_forest(mode = "classification") %>% 
  set_engine("ranger", importance = "impurity")

## rf workflow
rf_screen_workflow <- workflow() %>% 
  add_model(rf_screen) %>% 
  add_recipe(recipe_screen)

## fit model
set.seed(7)
screen_fit <- fit(rf_screen_workflow, class_include) 

## Create list of 50 best predictors from initial rf
best_pred <- screen_fit %>% 
  extract_fit_parsnip() %>% 
  vip::vi() %>% 
  head(57) %>% 
  filter(!(Variable %in% c("x335", "x249", "x383", "x059", "x302", 
                           "x390", "x213"))) %>% 
  select(Variable) %>% 
  unlist() %>% 
  unname() 

### examine highly correlated predictors among best 50 
### run, update selection to rm, and iterate until no very high correlation
class_train %>% 
  drop_na() %>% 
  select(best_pred) %>% 
  cor() %>% 
  as.table() %>% 
  as.data.frame() %>% 
  filter(Var1 != Var2) %>% 
  mutate(Freq = abs(Freq)) %>%
  arrange(desc(Freq)) %>% 
  filter(Freq > .95) %>%
  filter(seq_len(nrow(.)) %% 2 == 0) %>% 
  filter(!Var1 %in% c(Var2)) %>% 
  select(Var1) %>% 
  unlist() %>% 
  unique()

## Create list of 30 best vars for interactions
best_pred30 <- best_pred %>% head(30)

## Create list of 300 best predictors for PCA
best_pred300 <- screen_fit %>% 
  extract_fit_parsnip() %>% 
  vip::vi() %>% 
  head(300) %>% 
  select(Variable) %>% 
  unlist() %>% 
  unname() 

## list of vars with any missingness 
miss_train <- class_train %>% # missing in training
  select(all_of(best_pred)) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

## list of vars with any missingness in test
miss_test <- class_test %>% # missing in testing
  select(all_of(best_pred)) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

## combine
miss_combo <- c(miss_train, miss_test) %>% # missing in either test or train
  unique()

## list of vars with any missingness for PCA
miss_train300 <- class_train %>% # missing in training
  select(all_of(best_pred300)) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

## list of vars with any missingness in test for PCA
miss_test300 <- class_test %>% # missing in testing
  select(all_of(best_pred300)) %>% 
  skim() %>% 
  filter(n_missing != 0) %>% 
  select(skim_variable) %>% 
  unlist() %>% 
  unname()

## combine for PCA
miss_combo300 <- c(miss_train300, miss_test300) %>% # missing in either test or train
  unique()

# list of highly skewed predictors for YeoJohnson
skew_pred <- class_train %>% 
  select(all_of(best_pred300)) %>% 
  psych::describe() %>% 
  filter(skew > 1 | skew < -1) %>% 
  select(vars, skew, min, max) %>% 
  arrange(desc(skew)) %>% 
  rownames()

# set up folds
set.seed(902)
class_fold <- vfold_cv(class_train, v = 5, repeats = 3)

# set up main recipe
recipe_main <- recipe(y ~ ., data = class_train) %>% 
  step_rm(all_predictors(), -best_pred) %>% # recipe
  step_YeoJohnson(all_numeric_predictors()) %>% # transform to make more normal
  step_impute_knn(miss_combo) %>% # impute missing
  step_novel(all_nominal_predictors()) %>% # allow test to take on new levels
  step_dummy(all_nominal_predictors(), one_hot = T) %>% # turn nominal into dummies
  step_nzv(all_predictors(), unique_cut = 1) %>% # remove near zero variance
  step_normalize(all_predictors()) # scale and center

# set up interaction recipe
recipe_int <- recipe_main %>% 
  step_interact(~ c(best_pred30)^2) # add all two-way interactions between 30 best pred

# set up PCA recipe
recipe_pca <- recipe(y ~ ., data = class_train) %>% 
  step_rm(all_predictors(), -best_pred300) %>% # recipe
  step_YeoJohnson(skew_pred) %>% # transform to make more normal
  step_impute_knn(miss_combo300) %>% # impute missing
  step_novel(all_nominal_predictors()) %>% # allow test to take on new levels
  step_dummy(all_nominal_predictors(), one_hot = T) %>% # turn nominal into dummies
  step_pca(all_predictors(), -best_pred, num_comp = 80) %>%  # PCA for second tier preds
  step_nzv(all_predictors(), unique_cut = 1) %>% # remove near zero variance
  step_normalize(all_predictors()) %>% 
  step_interact(~ c(best_pred30)^2)

# save needed objects
save(best_pred, best_pred30, best_pred300, skew_pred, miss_combo, miss_combo300, 
     recipe_main, recipe_int, recipe_pca, class_fold,
     file = "results/modeling_objs.rda")
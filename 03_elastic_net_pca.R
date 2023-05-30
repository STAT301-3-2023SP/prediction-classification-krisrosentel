# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, glmnet)

# handle common conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up elastic net model
elastic_model <- logistic_reg(mixture = tune(), 
                              penalty = tune()) %>% 
  set_engine("glmnet")

# elastic net parameters
elastic_params <- extract_parameter_set_dials(elastic_model) %>% 
  update(penalty = penalty(range = c(-2.5, 0)),
         mixture = mixture(range = c(0, 1)))
elastic_grid <- grid_regular(elastic_params, levels = 6)

# prep recipe
recipe_pca <- recipe_pca %>% prep()

# elastic workflow
elastic_workflow <-  
  workflow() %>% 
  add_model(elastic_model) %>% 
  add_recipe(recipe_pca)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("Elastic Net")

## tune elastic
elastic_pca_tuned <- elastic_workflow %>% 
  tune_grid(class_fold, grid = elastic_grid,
            control = control_grid(save_pred = TRUE, 
                                   save_workflow = TRUE,
                                   parallel_over = "everything"))

# end parallel processing
stopCluster(cl)

# stop clock
toc(log = TRUE)
time_log <- tic.log(format = FALSE)

# save run time
elastic_pca_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = (time_log[[1]]$toc - time_log[[1]]$tic) 
)

# save
save(elastic_pca_tuned, elastic_pca_tictoc,  
          file = "results/elastic_pca_cv.rda")
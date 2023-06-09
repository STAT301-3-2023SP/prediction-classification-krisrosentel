# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, kernlab)

# handle common conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up svm model
svm_rad_model <- svm_rbf(
  mode = "classification", 
  cost = tune(),
  rbf_sigma = tune()) %>%
  set_engine("kernlab")

## svm parameters
svm_rad_params <- extract_parameter_set_dials(svm_rad_model) %>% 
  update(cost = cost(c(-2, 6)),
         rbf_sigma = rbf_sigma(c(-4, 0))) 
svm_rad_grid <- grid_regular(svm_rad_params, levels = 5)

# svm workflow
svm_rad_workflow <- workflow() %>% 
  add_model(svm_rad_model) %>% 
  add_recipe(recipe_int)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("SVM, Radial")

## tune svm rad
svm_rad_tuned <- svm_rad_workflow %>% 
  tune_grid(class_fold, grid = svm_rad_grid,
            control = control_grid(save_pred = TRUE, 
                                   save_workflow = TRUE,
                                   parallel_over = "everything"))

# end parallel processing
stopCluster(cl)

# stop clock
toc(log = TRUE)
time_log <- tic.log(format = FALSE)

# save run time
svm_rad_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(svm_rad_tuned, svm_rad_tictoc,  
          file = "results/svm_rad_cv.rda")
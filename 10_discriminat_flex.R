# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, earth, discrim, mda)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up discriminant model
disc_model <- discrim_flexible(
  mode = "classification",
  num_terms = tune(),
  prod_degree = tune()) %>%
  set_engine("earth")

## disc parameters
disc_params <- extract_parameter_set_dials(disc_model) %>% 
  update(num_terms = num_terms(c(5, 100)),
       prod_degree = prod_degree(c(1, 3))) 
disc_grid <- grid_regular(disc_params, levels = c(6, 3))

# disc workflow
disc_workflow <- workflow() %>% 
  add_model(disc_model) %>% 
  add_recipe(recipe_main)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("Discriminant Analysis, Flexible")

## tune disc
set.seed(91) # set seed 
disc_tuned <- disc_workflow %>% 
  tune_grid(class_fold, grid = disc_grid,
            control = control_grid(save_pred = TRUE, 
                                   save_workflow = TRUE,
                                   parallel_over = "everything"))

# end parallel processing
stopCluster(cl)

# stop clock
toc(log = TRUE)
time_log <- tic.log(format = FALSE)

# save run time
disc_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(disc_tuned, disc_tictoc,  
          file = "results/disc_cv.rda")
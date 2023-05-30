# load libraries
library(pacman)
p_load(tidymodels, tidyverse, doParallel, tictoc, earth, baguette)

# deal with package conflicts
tidymodels_prefer()

# load saved objects from setup
load("results/modeling_objs.rda")

# set up mars model
mars_model <- bag_mars(
  mode = "classification",
  num_terms = tune(),
  prod_degree = tune()) %>%
  set_engine("earth", times = 20)

## mars parameters
mars_params <- extract_parameter_set_dials(mars_model) %>% 
  update(num_terms = num_terms(c(5, 80)),
       prod_degree = prod_degree(c(1, 3))) 
mars_grid <- grid_regular(mars_params, levels = c(5, 3))

# mars workflow
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(recipe_main)

# Set up parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# start clock
tic.clearlog()
tic("MARS, Bagged")

## tune mars
set.seed(27) # set seed 
mars_tuned <- mars_workflow %>% 
  tune_grid(class_fold, grid = mars_grid,
            control = control_grid(save_pred = TRUE, 
                                   save_workflow = TRUE,
                                   parallel_over = "everything"))

# end parallel processing
stopCluster(cl)

# stop clock
toc(log = TRUE)
time_log <- tic.log(format = FALSE)

# save run time
mars_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

# save
save(mars_tuned, mars_tictoc,  
          file = "results/mars_cv.rda")
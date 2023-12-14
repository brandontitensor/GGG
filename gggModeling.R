#############
##LIBRARIES##
#############

library(tidymodels) 
library(tidyverse)
library(vroom) 
library(glmnet)
library(randomForest)
library(doParallel)
library(xgboost)
tidymodels_prefer()
conflicted::conflicts_prefer(yardstick::rmse)
library(rpart)
library(stacks)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(themis)
library(bonsai)
library(lightgbm)


####################
##WORK IN PARALLEL##
####################

all_cores <- parallel::detectCores(logical = FALSE)
#num_cores <- makePSOCKcluster(NUMBER OF CORES)
registerDoParallel(cores = all_cores)

#stopCluster(num_cores)

########
##DATA##
########

my_data <- vroom("train.csv")
test_data <- vroom("test.csv")
my_data$type <- as.factor(my_data$type)

# my_data_mv<- vroom("trainMV.csv")

#######
##EDA##
#######

GGally::ggpairs(my_data)

##########
##RECIPE##
##########

my_recipe <- recipe(type~., data=my_data)  %>%
  update_role(id, new_role="id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_mutate(bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul,
         hair_soul = hair_length * has_soul) %>%
  step_rm(c(color,id)) %>%
  step_range(all_numeric_predictors(), min=0, max=1) 

prepped_recipe <- prep(my_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)

rmse_vec(my_data[is.na(my_data_mv)],bake_1[is.na(my_data_mv)])

###################
##NEURAL NETWORKS##
###################

nn_recipe <- recipe(type~., data=my_data) %>%
  update_role(id, new_role="id") %>%
  step_mutate_at(color, fn = factor) %>% 
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1) 

prepped_recipe <- prep(nn_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50, #or 100 or 2507
                activation="relu") %>%
set_engine("keras", verbose=0) %>% 
  set_mode("classification")

# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50) %>%
#   set_engine("nnet") %>% 
#   set_mode("classification")


nn_workflow <- workflow() %>% #Creates a workflow
  add_recipe(nn_recipe) %>% #Adds in my recipe
  add_model(nn_model) 

tuning_grid_nn <- grid_regular(hidden_units(range=c(1,100)),
                            levels=3)
folds_nn <- vfold_cv(my_data, v = 5, repeats=1)

tuned_nn <- nn_workflow %>%
tune_grid(resamples=folds_nn,
          grid=tuning_grid_nn,
          metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                             precision, accuracy))

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune_nn <- tuned_nn %>%
  select_best("roc_auc")

final_rf_wf <- nn_workflow %>% 
  finalize_workflow(bestTune_nn) %>% 
  fit(data = my_data)


nn_predictions <- final_rf_wf %>% 
  predict(new_data = test_data, type="class")

nn_predictions <- bind_cols(test_data$id,nn_predictions$.pred_class)

colnames(nn_predictions) <- c("id","type")

nn_predictions <- as.data.frame(nn_predictions)

vroom_write(nn_predictions,"nn_predictions.csv",',')


############
##BOOSTING##
############

boost_model <- boost_tree(tree_depth=tune(),
                          trees= tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(boost_model) 

tuning_grid_boost <- grid_regular(tree_depth(),
                                  learn_rate(),
                                  trees(),
                                  levels = 5)
folds_boost <- vfold_cv(my_data, v = 10, repeats=1)

CV_results_boost <- boost_workflow %>%
  tune_grid(resamples=folds_boost,
            grid=tuning_grid_boost,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy))
bestTune_boost <- CV_results_boost %>%
  select_best("accuracy")

final_wf_boost <- boost_workflow %>% 
  finalize_workflow(bestTune_boost) %>% 
  fit(data = my_data)


boost_predictions <- final_wf_boost %>% 
  predict(new_data = test_data, type="class")


boost_predictions <- bind_cols(test_data$id,boost_predictions$.pred_class)

colnames(boost_predictions) <- c("id","type")

vroom_write(boost_predictions,"boost_predictions.csv",',')


########
##BART##
########

bart_model <- bart(trees = 1113,
                   prior_terminal_node_coef = tune(),
                   prior_terminal_node_expo = tune(),
                   prior_outcome_range = tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_workflow <- workflow() %>% #Creates a workflow
  add_recipe(nn_recipe) %>% #Adds in my recipe
  add_model(bart_model) 

tuning_grid_bart <- grid_regular(prior_terminal_node_coef(),
                                 prior_terminal_node_expo(),
                                 prior_outcome_range(),
                                 levels = 3)
folds_bart <- vfold_cv(my_data, v = 5, repeats=1)

CV_results_bart <- bart_workflow %>%
  tune_grid(resamples=folds_bart,
            grid=tuning_grid_bart,
            metrics=metric_set( f_meas, sens, recall, spec,
                              accuracy))
bestTune_bart <- CV_results_bart %>%
  select_best("accuracy")

final_bart_wf <- bart_workflow %>% 
  finalize_workflow(bestTune_bart) %>% 
  fit(data = my_data)


bart_predictions<- final_bart_wf %>% 
  predict(new_data = test_data)

bart_predictions <- final_bart_wf %>% 
  predict(new_data = test_data, type="class")


bart_predictions <- bind_cols(test_data$id,bart_predictions$.pred_class)

colnames(bart_predictions) <- c("id","type")

bart_predictions <- as.data.frame(bart_predictions)

vroom_write(bart_predictions,"bart_predictions.csv",',')

######
##NB##
######

nb_recipe <- recipe(type~., data=my_data)  %>%
  step_rm(color) %>%
  step_range(c(all_numeric_predictors(),-id), min=0, max=1) 

prepped_recipe <- prep(nb_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)

NB_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

NB_workflow <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(NB_model)



tuning_grid_NB <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 10)
folds_NB <- vfold_cv(my_data, v = 20, repeats=10)

CV_results_NB <- NB_workflow %>%
  tune_grid(resamples=folds_NB,
            grid=tuning_grid_NB,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy))

# Evaluate the model
nb_metrics <- metric_set(roc_auc, f_meas, sens, recall, spec, precision, accuracy)
nb_results <- CV_results_NB %>%
              collect_metrics() %>%
              filter(.metrics %in% nb_metrics)
collect_metrics(CV_results_NB)

bestTune_NB <- CV_results_NB %>%
  select_best("accuracy")

final_NB_wf <- NB_workflow %>% 
  finalize_workflow(bestTune_NB) %>% 
  fit(data = my_data)


extract_fit_engine(final_NB_wf) %>% 
  summary()

NB_predictions <- final_NB_wf %>% 
  predict(NB_workflow, new_data=test_data, type="class")

NB_predictions <- bind_cols(test_data$id,NB_predictions$.pred_class)

colnames(NB_predictions) <- c("id","type")

vroom_write(NB_predictions,"NB_predictions.csv",',')


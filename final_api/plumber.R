# Find out more about building APIs with Plumber here:
#
#    https://www.rplumber.io/
#

# Read in data and fit best model to the entire data set
library(readr)
library(dplyr)
library(janitor)
library(ranger)
library(yardstick)
library(tidymodels)

diabetes_data <- read_csv("../diabetes_binary_health_indicators_BRFSS2015.csv")

diabetes_data <- diabetes_data |>
  mutate(
    Diabetes_binary = factor(Diabetes_binary, 
                             levels = c(0, 1),
                             labels = c("No diabetes",
                                        "Prediabetes/Diabetes")),
    HighBP = factor(HighBP,
                    levels = c(0, 1),
                    labels = c("Not high BP", "High BP")),
    HighChol = factor(HighChol,
                      levels = c(0, 1),
                      labels = c("Not high cholesterol", "High cholesterol")), 
    Smoker = factor(Smoker,
                    levels = c(0, 1),
                    labels = c("Non-smoker", "Smoker")),
    Stroke = factor(Stroke,
                    levels = c(0, 1),
                    labels = c("No stroke", "Stroke")),
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack,
                                  levels = c(0, 1),
                                  labels = c("No heart disease/attack", 
                                             "Heart disease/attack")),
    PhysActivity = factor(PhysActivity,
                          levels = c(0, 1),
                          labels = c("No physical activity", "Physical activity")),
    HvyAlcoholConsump = factor(HvyAlcoholConsump,
                               levels = c(0, 1),
                               labels = c("Not a heavy drinker", "Heavy drinker")),
    AnyHealthcare = factor(AnyHealthcare,
                           levels = c(0, 1),
                           labels = c("No healthcare", "Healthcare")),
    GenHlth = factor(GenHlth,
                     levels = c(1, 2, 3, 4, 5),
                     labels = c("Excellent", "Very good", 
                                "Good", "Fair", "Poor")),
    Sex = factor(Sex,
                 levels = c(0, 1),
                 labels = c("Female", "Male")),
    Age = factor(Age,
                 levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                 labels = c("18-24", "25-29", "30-34", "35-39", "40-44",
                            "45-49", "50-54", "55-59", "60-64", "65-69",
                            "70-74", "75-79", "80 or older"))
  ) |>
  select(-CholCheck, -NoDocbcCost, -Fruits, -Veggies, -Income, -Education, 
         -DiffWalk) |>
  clean_names("snake")

set.seed(123)
diabetes_split <- initial_split(diabetes_data, prop = .70)
diabetes_train <- training(diabetes_split)

diabetes_recipe <- recipe(diabetes_binary ~ ., data = diabetes_train) |>
  update_role(age, new_role = "ID") |>
  step_rm(phys_hlth, ment_hlth, any_healthcare) |>
  step_dummy(high_bp, high_chol, smoker, stroke, phys_activity,
             heart_diseaseor_attack, hvy_alcohol_consump, sex, gen_hlth) |>
  step_normalize(bmi)

forest_model <- rand_forest(mtry = 4, trees = 100) |>
  set_engine("ranger") |>
  set_mode("classification")

forest_workflow <- workflow() |>
  add_recipe(diabetes_recipe) |>
  add_model(forest_model)

forest_fit <- forest_workflow |>
  fit(diabetes_data)



library(plumber)

#* @apiTitle Diabetes Health Indicators API
#* @apiDescription The Diabetes Health Indicators API has three endpoints. The pred endpoint will take in a predictor from the Diabetes Health Indicators data set, and return the mean of that variable (if numeric) or the most prevalent class (if categorical). The info endpoint will provide the author's name and the URL for the EDA and Modeling of the data. The confusion endpoint will display an image of the confusion matrix for the final model fit, a random forest model for the response variable, diabetes status (no diabetes or diabetes/pre-diabetes). 

#* Info endpoint
#* @get /info
function() {
    list(Author = "Kim Costello", 
         URL = "https://kimacostello.github.io/ST558_Final_Project/EDA.html") 
}

#* Predictor variables
#* @param var
#* @get /pred
function(var) {
  column <- diabetes_data[[var]]
  if (is.numeric(column)) {return(mean(column))}
  
  else {uni_var <- unique(column)
  return(uni_var[which.max(tabulate(match(column, uni_var)))])
  }
}
# Three example function calls:
# http://127.0.0.1:8218/pred?var=bmi
# http://127.0.0.1:8218/pred?var=high_bp
# http://127.0.0.1:8218/pred?var=smoker

#* Confusion Matrix
#* @get /confusion
#* @serializer png
function() {
  predictions <- predict(forest_fit, new_data = diabetes_data)
  
  data_with_predictions <- diabetes_data |>
    bind_cols(predictions)
  
  cm <- conf_mat(data_with_predictions, truth = diabetes_binary, estimate = .pred_class)
  
  print(autoplot(cm, type = "heatmap"))
  }




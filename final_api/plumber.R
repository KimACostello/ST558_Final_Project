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
library(tibble)

diabetes_data <- read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

diabetes_data <- diabetes_data |>
  mutate(
    Diabetes_binary = factor(Diabetes_binary, 
                             levels = c(0, 1),
                             labels = c("No diabetes",
                                        "Prediabetes/Diabetes")),
    HighBP = factor(HighBP,
                    levels = c(0, 1),
                    labels = c("no", "yes")),
    HighChol = factor(HighChol,
                      levels = c(0, 1),
                      labels = c("no", "yes")), 
    Smoker = factor(Smoker,
                    levels = c(0, 1),
                    labels = c("no", "yes")),
    Stroke = factor(Stroke,
                    levels = c(0, 1),
                    labels = c("no", "yes")),
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack,
                                  levels = c(0, 1),
                                  labels = c("no", 
                                             "yes")),
    PhysActivity = factor(PhysActivity,
                          levels = c(0, 1),
                          labels = c("no", "yes")),
    HvyAlcoholConsump = factor(HvyAlcoholConsump,
                               levels = c(0, 1),
                               labels = c("no", "yes")),
    GenHlth = factor(GenHlth,
                     levels = c(1, 2, 3, 4, 5),
                     labels = c("excellent", "very good", 
                                "good", "fair", "poor")),
    Sex = factor(Sex,
                 levels = c(0, 1),
                 labels = c("female", "male"))
  ) |>
  select(-CholCheck, -NoDocbcCost, -Fruits, -Veggies, -Income, -Education, 
         -DiffWalk, -AnyHealthcare, -MentHlth, -PhysHlth, -Age) |>
  clean_names("snake")


set.seed(123)
diabetes_split <- initial_split(diabetes_data, prop = .70)
diabetes_train <- training(diabetes_split)

diabetes_recipe <- recipe(diabetes_binary ~ ., data = diabetes_train) |>
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
#* @apiDescription The Diabetes Health Indicators API has three endpoints. The info endpoint will provide the author's name and the URL for the EDA and Modeling of the data.The pred endpoint will take in values for any of the predictor variables and make a prediction on the diabetes status (no diabetes or diabetes/pre-diabetes) based on our model. For any predictor variable values that are not given, the mean or most prevalent class for that variable will be used. The confusion endpoint will display an image of the confusion matrix for the final model fit, a random forest model for the response variable, diabetes status (no diabetes or diabetes/pre-diabetes). 

#* Info
#* @get /info
function() {
    list(Author = "Kim Costello", 
         URL = "https://kimacostello.github.io/ST558_Final_Project/EDA.html") 
}


#* Obtain Predictions
#* @param high_bp yes or no
#* @param high_chol yes or no
#* @param bmi Body Mass Index value
#* @param smoker yes or no
#* @param stroke yes or no
#* @param heart_diseaseor_attack yes or no
#* @param phys_activity yes or no
#* @param hvy_alcohol_consump No or Yes
#* @param gen_hlth excellent, very good, good, fair, or poor
#* @param sex female or male
#* @get /pred
function(high_bp = bp_mode, 
         high_chol = chol_mode, 
         bmi = bmi_mean, 
         smoker = smoker_mode, 
         stroke = stroke_mode, 
         heart_diseaseor_attack = heart_disease_mode, 
         phys_activity = phys_activity_mode, 
         hvy_alcohol_consump = hvy_alcohol_mode, 
         gen_hlth = gen_hlth_mode, 
         sex = sex_mode) {
  
  #Function for the statistical modes to use in the pred endpoint since the mode() function in r returns the storage type. 
  get_mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  
  bp_mode = get_mode(diabetes_data$high_bp) 
  chol_mode = get_mode(diabetes_data$high_chol) 
  bmi_mean = mean(diabetes_data$bmi) 
  smoker_mode = get_mode(diabetes_data$smoker) 
  stroke_mode = get_mode(diabetes_data$stroke) 
  heart_disease_mode = get_mode(diabetes_data$heart_diseaseor_attack) 
  phys_activity_mode = get_mode(diabetes_data$phys_activity) 
  hvy_alcohol_mode = get_mode(diabetes_data$hvy_alcohol_consump) 
  gen_hlth_mode = get_mode(diabetes_data$gen_hlth) 
  sex_mode = get_mode(diabetes_data$sex)

  new_data <- tibble(high_bp = factor(high_bp, levels = c("no", "yes")),
                     high_chol = factor(high_chol, levels = c("no", "yes")),
                     bmi = as.numeric(bmi),
                     smoker = factor(smoker, levels = c("no", "yes")),
                     stroke = factor(stroke, levels = c("no", "yes")),
                     heart_diseaseor_attack = factor(heart_diseaseor_attack, 
                                                     levels = c("no", "yes")),
                     phys_activity = factor(phys_activity, levels = c("no", 
                                                                      "yes")),
                     hvy_alcohol_consump = factor(hvy_alcohol_consump, 
                                                  levels = c("no", "yes")),
                     gen_hlth = factor(gen_hlth, levels = c("excellent", 
                                                            "very good", 
                                                            "good", "fair", 
                                                            "poor")),
                     sex = factor(sex, levels = c("female", "male")))
  
  predict(forest_fit, new_data = new_data)
  
  }


# Three example function calls:
# http://127.0.0.1:39458/pred?high_bp=yes&high_chol=yes&bmi=35&smoker=yes&stroke=yes&heart_diseaseor_attack=yes&phys_activity=no&hvy_alcohol_consump=yes&gen_hlth=poor&sex=female
# http://127.0.0.1:39458/pred?high_bp=no&smoker=no&stroke=yes&heart_diseaseor_attack=yes&phys_activity=no&hvy_alcohol_consump=yes&gen_hlth=poor&sex=female
# http://127.0.0.1:39458/pred?high_bp=no&bmi=40&smoker=no&phys_activity=no&hvy_alcohol_consump=yes&gen_hlth=excellent&sex=male

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




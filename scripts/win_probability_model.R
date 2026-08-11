# setup

library(baseballr)
library(tidymodels)
library(xgboost)

#1.pull schedule

schedule_raw <- mlb_schedule(season = 2025)

schedule <- schedule_raw |> 
  mutate(game_date = as.Date(game_date)) |> #change date from character variable  
  filter(
    game_type == "R", #reg season 
    game_date >= as.Date("2025-04-01"),
    game_date <= as.Date("2025-04-30")
  )

game_pks <- unique(schedule$game_pk)

#2.pull play by play

safe_pbp <- safely(mlb_pbp)

pbp_list <- map(game_pks, function(pk) {
  Sys.sleep(0.2)
  safe_pbp(pk)$result
})

is_valid_df <- map_lgl(pbp_list, is.data.frame) #making sure each entry is a data frame 

dropped_games <- game_pks[!is_valid_df]
length(dropped_games)

pbp_list <- pbp_list[is_valid_df]
names(pbp_list) <- game_pks[is_valid_df]

pbp_raw <- bind_rows(pbp_list, .id = "game_pk")

saveRDS(pbp_raw, "data/pbp_raw_2025_04.rds") #save pbp data 

glimpse(pbp_raw)

#3. Feature engineering

pbp <- pbp_raw |> 
  filter(last.pitch.of.ab == "true") |> 
  transmute(
    game_pk = as.integer(game_pk),
    at_bat_index = about.atBatIndex,
    inning = about.inning,
    is_top = about.isTopInning,
    outs = count.outs.start,
    home_score = result.homeScore,
    away_score = result.awayScore,
    score_diff = home_score - away_score,
    on_1b = !is.na(matchup.postOnFirst.id),
    on_2b = !is.na(matchup.postOnSecond.id),
    on_3b = !is.na(matchup.postOnThird.id)
  ) |> 
  mutate(
    base_state = paste0(
      ifelse(on_1b, "1", "0"),
      ifelse(on_2b, "1", "0"),
      ifelse(on_3b, "1", "0")
    )
  )

#4 label each row with eventual outcome 

game_outcomes <-pbp |> 
  group_by(game_pk) |> 
  arrange(at_bat_index, .by_group = TRUE) |> 
  slice_tail(n = 1) |> 
  transmute(game_pk, home_win = as.integer(home_score > away_score)) |>  
  ungroup() 

model_data <- pbp |> 
  left_join(game_outcomes, by = "game_pk") |> 
  filter(!is.na(home_win)) |> 
  mutate(
    is_top = as.integer(is_top),
    base_state = factor(base_state),
    home_win = factor(home_win, levels = c(0,1))
  ) |> 
  drop_na(inning, outs, score_diff, base_state, home_win)

# sanity checks
table(model_data$home_win)
summary(model_data$score_diff)

#5. train/test split by game 

set.seed(67)
game_ids <- unique(model_data$game_pk)
train_ids <- sample(game_ids, size = floor(0.8 * length(game_ids)))

train_data <- model_data |> filter(game_pk %in% train_ids)
test_data <- model_data |> filter(!game_pk %in% train_ids)

#6. Shared recipe 

base_recipe <- recipe(
  home_win ~ inning + is_top + base_state + outs + score_diff, 
  data = train_data
) |> 
  step_dummy(base_state)

#7 baseline logit regression 

logit_spec <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")

logit_wf <- workflow() |> 
  add_recipe(base_recipe) |> 
  add_model(logit_spec)

logit_fit <- fit(logit_wf, data = train_data)

logit_interact <- glm(
  home_win ~ inning + is_top * base_state + outs + score_diff,
  data = train_data,
  family = binomial()
)

summary(logit_interact)

logit_baseline <- glm(
  home_win ~ inning + is_top + outs + score_diff + base_state,
  data = train_data,
  family = binomial()
)

AIC(logit_baseline, logit_interact)

#8. XGBoost

xgb_spec <- boost_tree(
  trees = 150,
  tree_depth = 4,
  learn_rate = 0.1
) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

xgb_wf <- workflow() |> 
  add_recipe(base_recipe) |> 
  add_model(xgb_spec)

xgb_fit <- fit(xgb_wf, data = train_data)

#9. generate predictions 

# type = "prob" gives .pred_0 / .pred_1 -- .pred_1 is P(home_win = 1)

test_preds <- test_data |> 
  bind_cols(
    predict(logit_fit, new_data = test_data, type = "prob") |> 
      select(logit_pred = .pred_1)
  ) |> 
  bind_cols(
    predict(xgb_fit, new_data = test_data, type = "prob") |> 
      select(xgb_pred = .pred_1)
  ) 

test_preds$logit_interact_pred <- predict.glm(logit_interact, newdata = test_data, type = "response")

#10. calibration check

calibration_data <- test_preds |> 
  mutate(home_win_num = as.numeric(as.character(home_win))) |> 
  select(home_win_num, logit_pred, xgb_pred, logit_interact_pred) |> 
  pivot_longer(cols = c(logit_pred, xgb_pred, logit_interact_pred), 
               names_to = "model", values_to = "pred") |> 
  mutate(pred_bin = cut(pred, breaks = seq(0,1,by = 0.05))) |> 
  group_by(model, pred_bin) |> 
  summarize(
    mean_pred = mean(pred),
    actual_win_rate = mean(home_win_num),
    n = n(),
    .groups = "drop"
  )

ggplot(calibration_data, aes(x = mean_pred, y = actual_win_rate, size = n, color = model)) + 
  geom_point(alpha = 0.7) + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") + 
  labs(
    title = "Win Probability Model Calibration",
    x = "Predicted Probability (Home Win)", 
    y = "Observed Win Rate", 
    size = "N plays",
    color = "model"
  ) + 
  xlim(0,1) + ylim(0,1) + 
  theme_minimal() + 
  facet_wrap(~model)

#11 Log_loss comparison 

log_loss <- function(actual, predicted) {
  eps <- 1e-15
  predicted <- pmin(pmax(predicted,eps), 1 - eps)
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}

test_label <- as.numeric(as.character(test_preds$home_win))

cat("Logistic regression (baseline)   log-loss:", log_loss(test_label, test_preds$logit_pred), "\n")
cat("Logistic regression (interaction) log-loss:", log_loss(test_label, test_preds$logit_interact_pred), "\n")
cat("XGBoost                          log-loss:", log_loss(test_label, test_preds$xgb_pred), "\n")
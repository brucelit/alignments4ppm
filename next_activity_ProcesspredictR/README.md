To run next activity prediction using ProcesspredictR, first install R and RStudio following the [tutorial](https://rstudio-education.github.io/hopr/starting.html). Then, install BupaR to use processpredictR following the [tutorial](https://bupaverse.github.io/docs/install.html). The code below is adapted based on prediction workflow from [Bupar](https://bupaverse.github.io/docs/predict_workflow.html). To run the event log split and further experiments in RStudio, change 'bpi17_offer.xes' into your event log for function read_xes in line 6:
```
library(tensorflow)
library(xesreadR)
library(processpredictR)
library(bupaverse)
library(dplyr)
data <- read_xes("logs/bpi17_offer.xes")
df <- prepare_examples(data, task = "next_activity")
df$start_time <- as.POSIXct(df$start_time)

# Sort by start_time in ascending order
df <- df[order(df$start_time)]

# Split into train and test (e.g., 80% train, 20% test)
split_index <- floor(0.8 * nrow(df))
train_df <- df[1:split_index, ]
test_df <- df[(split_index + 1):nrow(df), ]

# Start training
model <- train_df %>% create_model(name = "my_model") 
model %>% compile() # model compilation
hist <- fit(object = model, train_data = train_df, epochs = 5)
predictions <- model %>% predict(test_data = test_df, output = "append")

# Returns loss and metrics
model %>% evaluate(test_df)
```
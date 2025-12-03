# Explainable Predictive Process Monitoring with Stochastic Alignments
## Introduction
This repository provides the implementation details and supplementary materials to support the findings presented in the paper "Explainable Predictive Process Monitoring with Stochastic Alignments".

## Dataset
**\dataset** folder contains the event logs used in this study. The event logs were sorted by time and chronologically divided, with the first 80% of the traces used as the training set and the remaining 20% as the test set. 

## Impelemented approach
The technique can be used as a standalone tool from the command line. 
### Single trace prediction
To predict the next activity of a prefix <Insert ticket, Assign seriousness, Take in charge ticket>, run the following command:
```
cargo run prediction pretra ./testlogs/helpdesk_80.xes "Insert ticket"
"Assign seriousness" "Take in charge ticket"
```

To predict the next activity of a prefix <Insert ticket, Assign seriousness, Take in charge ticket>, run the following command:
```
cargo run prediction presfx ./testlogs/helpdesk_80.xes "Insert ticket"
"Assign seriousness" "Take in charge ticket"
```

### Prediction experiments
To evaluate the performance of next activity prediction, run the following command:
 ```
cargo run prediction prenext ./testlogs/helpdesk_80.xes ./testlogs/helpdesk_20.xes
 ```

To evaluate the performance of suffix prediction, run the following command:
 ```
cargo run prediction presfx ./testlogs/helpdesk_80.xes ./testlogs/helpdesk_20.xes
 ```


## Other approaches used during evaluation
The following table includes all techniques used for evaluation.

| Name | Prediction task | Link |
|    :----:   | :----:   |          :---: | 
| LSTM   |  Next activity|  [Paper](https://link.springer.com/chapter/10.1007/978-3-319-59536-8_30), [Code](https://github.com/verenich/ProcessSequencePrediction?tab=readme-ov-file)  |
| ImagePP-Miner| Next activity|[Paper](https://ieeexplore.ieee.org/document/8786066), [Code](https://github.com/vinspdb/ImagePPMiner)|
|ProcessTransformer | Next activity|[Paper](https://arxiv.org/abs/2104.00721), [Code](https://github.com/Zaharah/processtransformer.) |      
|SEPHIGRAPH|Next activity|[Paper](https://link.springer.com/chapter/10.1007/978-3-031-92474-3_22), [Code](https://github.com/sebdisdv/SEPHIGRAPH)|
|ProcesspredictR|Next activity|[Paper](https://ceur-ws.org/Vol-3648/paper_972.pdf), [Code](https://bupaverse.github.io/docs/predict_workflow.html)|
|Transition system|Suffix|[Paper](), [Code]()|
|SuTraN|Suffix|[Paper](), [Code]()|
|CRTP-LSTM|Suffix|[Paper](), [Code]()|
|ASTON|Suffix|[Paper](), [Code]()|
|DOGE|Suffix|[Paper](), [Code]()|

In **\other_technqiues** folder, we upload the code support to run other techniques.

## Run the experiments
### ProcesspredictR
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

### LSTM
Run the experiments with the following command:
```
python train.py --fold_dataset ../data/[FOLD_DATASET] --full_dataset ../data/[FULL_DATASET] --train --test
```
For instance,:
```
python train.py --fold_dataset ../data/fold0_variation0_bpi17_offer.csv --full_dataset ../data/bpi17_offer.csv --test --train

```


### ImagePP-Miner
Run the experiments with the following command:
```
python run.py --full_dataset dataset/[FULL_DATASET].csv --fold_dataset dataset/[FOLD_DATASET].csv --train --test
```

For instance:
```
python run.py --full_dataset dataset/bpi17_offer.csv --fold_dataset dataset/fold0_variation0_bpi17_offer.csv --train --test
```

### Processtransformer
First, create the environment as follows:
```
conda create -n "zarahah" python=3.9
conda activate zarahah
conda install -c anaconda tensorflow-gpu=2.4.1
python -m pip install pandas==1.3.4 scikit-learn==1.0.1
```

After changing 'bpi17_offer' into the target event log, run the preprocessing as follows:
```
python data_processing.py --raw_log_file ./datasets/bpi17_offer/fold0_variation0_bpi17_offer.csv --task next_activity --dir_path ./datasets/bpi17_offer/ --dataset bpi17_offer
```


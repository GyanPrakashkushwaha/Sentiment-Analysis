# Sentiment-Analysis

## Web Demo


https://github.com/GyanPrakashkushwaha/Sentiment-Analysis/assets/127115588/249dd5b6-69d8-4b83-9fd2-2858ce8c648c



## About

This Project has made using Keras. Shows sentiment(Positive or negative). Trained in 50k IMDB reviews Data.



## Cloning and Running
To run this project locally, follow these steps:

1. Clone the repository:
   
```python
git clone https://github.com/GyanPrakashkushwaha/Sentiment-Analysis.git
```

2. Navigate to the project directory:

```python
cd Sentiment-Analysis
```

3. Create virtaul environment and activate it.
```python
virtalenv Sentiment-Analysis 
Sentiment-Analysis/Scipts/activate.ps1
```

4. Install packages:
```python
pip install -r requirments.txt
```

5.run the files in sequence:
- text preprocessing
- data preprocessing for training
- model training (from this file you will have SentimentAnalysis Model)
- using model
<br><br>
---

## (Model Architecture)
```python

model_features = 100 # for embedding layer
input_len = 150
model = keras.Sequential(name='LSTM_model')

model.add(Embedding(
    input_dim=56942,
    output_dim=model_features, input_length=input_len,name = 'input_layer'
))
model.add(Bidirectional(
    LSTM(units=64,activation=relu,return_sequences=True),
    name='LSTM_1'
))
model.add(
    Dropout(rate=0.5))
model.add(
    BatchNormalization())
model.add(
    Dropout(rate=0.5))
model.add(Bidirectional(
    LSTM(units=32,activation=relu,return_sequences=False),
    name='LSTM_2'
))
model.add(Dense(
    units=128,activation=relu,name='fully_connected_layer'
))
model.add(Dense(
    units=1,activation=sigmoid,name='output_layer'
))

def lr_schedule(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * np.exp(-0.1)

# learning rate scheduler callback to descrese the learning rate gradually as the epochs increases So that my alogrithm could not jump out of Global minima.
lr_scheduler = LearningRateScheduler(lr_schedule)

# Early stopping to stop the Neural Network when we get same Validation accuracy
early_stopping = EarlyStopping(
    monitor="accuracy",
    min_delta=0.00001,
    patience=5,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)

model.compile(optimizer=optimizer, # Used Adam because this has not any major disadvantages with custom learning rate because the convergence was very unstable.
               loss=binary_crossentropy, # because solving the classification problem
                 metrics=['accuracy'])  # I don't need to write about this you know.

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20,
                    batch_size=32, # I had tried different batch sizes but this has given my best results
                      callbacks=[lr_scheduler, early_stopping]) # these to prevent the NN from overfitting and scheduling learning rate to get optimum solution.
```


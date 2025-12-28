import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# File paths
TRAIN_CSV = 'ttan_train.csv'
TEST_CSV  = 'ttan_eval.csv'

# 1. Load Data
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# 2. Drop Any Rows with Missing Features or Labels
FEATURE_COLS = [
    'sex', 'age', 'n_siblings_spouses', 'parch',
    'fare', 'class', 'deck', 'embark_town', 'alone'
]

train_df = train_df.dropna(subset=FEATURE_COLS + ['survived'])
test_df  = test_df.dropna(subset=FEATURE_COLS + ['survived'])

# 3. Feature Encoding
X_train = pd.get_dummies(train_df[FEATURE_COLS], drop_first=True)
X_test  = pd.get_dummies(test_df[FEATURE_COLS],  drop_first=True)

# Align train/test columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train = train_df['survived'].values
y_test  = test_df['survived'].values

# 4. Build & Compile Model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train
model.fit(X_train, y_train,
          validation_split=0.2,
          epochs=200, #originally 50.
          batch_size=32,
          verbose=2)

# 6. Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss:.4f}  |  Test accuracy: {accuracy:.4f}')

# 7. Predict Probabilities
pred_probs = model.predict(X_test).flatten()

# 8. Plot Histogram of Predicted Probabilities
plt.figure(figsize=(10, 6))
plt.hist(pred_probs, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Predicted Survival Probability')
plt.ylabel('Count of Passengers')
plt.title('Histogram of Titanic Predicted Survival Probabilities')
plt.grid(axis='y', alpha=0.3)
plt.show()

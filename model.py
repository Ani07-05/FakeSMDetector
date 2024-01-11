import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import numpy as np

# Read the CSV file and stores it in a Pandas DataFrame called df
df = pd.read_csv(r"D:\ANN\x1k.csv")
df['is_fake'] = -1

# Define the detection logic    
def detect_fake_profile(row):
    if row['followersCount'] > 10000:
        return 0  # Real Profile
    elif row['friendsCount'] / (row['followersCount'] + 1) < 0.1:
        return 0  # Real Profile
    elif row['tweetsCount'] > 500:
        return 0  # Real Profile
    elif len(str(row['name'])) > 10:
        return 0  # Real Profile
    elif pd.notna(row['bio']) and len(str(row['bio'])) > 50:
        return 0  # Real Profile
    elif row['friendsCount'] > 500:
        return 0  # Real Profile
    elif len(str(row['screenName'])) > 5:
        return 0  # Real Profile
    elif row['tweetsCount'] / (row['followersCount'] + 1) > 0.5:
        return 0  # Real Profile
    else:
        return 1  # Fake Profile 







# Apply the detection logic
df['is_fake'] = df.apply(detect_fake_profile, axis=1)

# ... (Previous code)

# Drop non-numeric columns 
df_numeric = df[['followersCount', 'tweetsCount', 'friendsCount']]

# Normalize numerical values
scaler = StandardScaler()
df_numeric = scaler.fit_transform(df_numeric)
df[['followersCount', 'tweetsCount', 'friendsCount']] = df_numeric

# Handle categorical data
categorical_columns = ['screenName', 'name', 'bio']
df = pd.get_dummies(df, columns=categorical_columns)

# Define features and labels
x = df.drop(columns=['is_fake'])
y = df['is_fake']

# ... (Continue with the rest of your code)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to float32
x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)

# Build the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=50, validation_data=(x_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)



# Assuming 'new_data' is a DataFrame with the same format as the training data
# Drop non-numeric columns for StandardScaler
new_data = pd.read_csv(r"D:\ANN\newds.csv")
new_numeric = new_data[['followersCount', 'tweetsCount', 'friendsCount']]

# Normalize numerical values using the same scaler

new_numeric = scaler.transform(new_numeric)
new_data[['followersCount', 'tweetsCount', 'friendsCount']] = new_numeric

# Handle categorical data
new_data = pd.get_dummies(new_data, columns=categorical_columns)

# Make sure the columns are in the same order as in the training data

new_features = new_data[x.columns]

# Convert data to float32
new_features = np.float32(new_features)

# Predict using the model
predictions = model.predict(new_features)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_and_save_model(path):
    model = Sequential([
        Dense(128, input_dim=5000, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create dummy data to fit the model
    import numpy as np
    X_dummy = np.random.random((10, 5000))
    y_dummy = np.random.randint(2, size=(10, 1))
    
    # Fit the model on the dummy data
    model.fit(X_dummy, y_dummy, epochs=1)
    
    # Save the model weights
    model.save_weights(path)
    print(f"Model weights saved to {path}")

if __name__ == "__main__":
    create_and_save_model('models/model.weights.h5')

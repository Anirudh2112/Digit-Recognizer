import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

def load_and_preprocess_data():
    print("Loading training data...")
    train_data = pd.read_csv("digit-recognizer/train.csv")
    
    # Separate features and labels
    X = train_data.drop('label', axis=1).values
    y = train_data['label'].values
    
    # Reshape data for CNN input 
    X = X.reshape(-1, 28, 28, 1)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    return X, y

def visualize_sample(X, y, sample_size=5):
    """Visualize sample digits from the dataset"""
    fig, axes = plt.subplots(1, sample_size, figsize=(10, 2))
    for i in range(sample_size):
        axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Digit: {y[i]}')
    plt.tight_layout()
    plt.show()

def create_model():
    model = keras.Sequential([
        # Convolutional layers
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = create_model()
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model

def generate_predictions(model):
    print("Loading test data...")
    test_data = pd.read_csv("digit-recognizer/test.csv")
    
    X_test = test_data.values.reshape(-1, 28, 28, 1)
    X_test = X_test / 255.0
    
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

def main():
    X, y = load_and_preprocess_data()
    visualize_sample(X, y)
    model = train_model(X, y)
    generate_predictions(model)

if __name__ == "__main__":
    main()

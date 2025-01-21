# Digit-Recognizer

# 🔢 MNIST Digit Recognition with CNN

Hey there! 👋 This is my first project on Kaggle and I decided to take on the classic MNIST digit recognition challenge. I've built a Convolutional Neural Network (CNN) using Keras that can recognize handwritten digits with pretty impressive accuracy.

## 🎯 What's This All About?

The MNIST dataset is like the "Hello World" of machine learning - it's got tons of handwritten digits (0-9) that we can use to teach our model to recognize numbers. Each image is 28x28 pixels in grayscale. Think of it as teaching a computer to read numbers like a human would!

## 🚀 Features

* Built with Keras/TensorFlow (because life's too short for implementing CNNs from scratch!)
* Uses a CNN architecture (because if you're doing image stuff, CNNs are your best friend)
* Includes data visualization (so you can actually see what you're working with)
* Training history plots (to make sure we're not overfitting)
* Generates submission file in the competition format (keeping it clean and simple)

## 🛠️ Model Architecture

Here's what's under the hood:

```python
Input → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Dropout → Output(10)
```

Why this architecture? It's simple enough to train quickly but complex enough to get good accuracy. The convolutional layers pick up on the patterns, max pooling helps with efficiency, and dropout keeps our model from memorizing the training data.

## 📦 Requirements

```python
tensorflow
pandas
numpy
matplotlib
scikit-learn
```

## 🏃‍♂️ How to Run

1. Clone this repo
2. Drop your `train.csv` and `test.csv` files in the project directory
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python recognizer.py
   ```
5. Grab a snack while it trains
6. Check out the `submission.csv` file once done!

## 📈 What to Expect

* Training takes about 5-10 minutes on a decent CPU
* Validation accuracy should hit around 98-99%
* You'll get some nice plots showing the training progress
* The submission file will be ready for the competition

## 💡 Tips

* Feel free to play with the hyperparameters (learning rate, batch size, etc.)
* Try adding more convolutional layers if you want to push the accuracy higher
* Data augmentation could help if you're feeling adventurous

## 📝 To-Do/Future Scope

- [ ] Add data augmentation
- [ ] Experiment with different architectures
- [ ] Add model checkpointing
- [ ] Implement cross-validation

## 🤝 Contributing

Found a bug? Have a cool idea? Feel free to open an issue or submit a PR. Let's make this even better together!

## 📜 License

MIT License - go wild! Just remember to give credit where it's due.

## 🙌 Acknowledgments

Shoutout to Yann LeCun and the MNIST dataset creators. Without them, we'd probably still be practicing machine learning on cat pictures (not that there's anything wrong with that! 😺).

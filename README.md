# Career Guidance Chatbot

This project is a **Career Guidance Chatbot** designed to help users explore educational courses and career paths 
across different domains such as commerce, science, arts, engineering, and medical fields. 
The chatbot leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to understand user queries and provide relevant, real-time responses.
The backend is developed using **Flask**, while the machine learning model is trained using deep learning techniques.

## Features
- Interactive chatbot for career guidance.
- Supports queries in domains like commerce, science, engineering, and medical fields.
- Trained using a **deep neural network** for intent classification.
- Real-time responses based on user input.
- Scalable architecture for adding more intents and responses.

## Technologies Used
- **Python**: Primary programming language.
- **Flask**: Backend framework to deploy the chatbot.
- **Keras**: Deep learning framework for model training.
- **Natural Language Toolkit (NLTK)**: For text processing (tokenization, lemmatization).
- **JSON**: Storing predefined intents and patterns.
- **Stochastic Gradient Descent (SGD)**: Optimizer for training the model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/career-guidance-chatbot.git


Install the dependencies:

Copy code
pip install -r requirements.txt

Download NLTK dependencies:
Copy code
import nltk
nltk.download('punkt')
nltk.download('wordnet')

How to Use

Train the Model: Run the training.py script to train the chatbot's model.

Copy code
python training.py
Start the Chatbot: Run the Flask application to start the chatbot server:

Copy code
python app.py
The server will start on http://127.0.0.1:5000.

How It Works

Data Preprocessing: The chatbot uses Bag of Words for text feature extraction, converting user inputs into a numerical format.
Model Training:

A Deep Neural Network with multiple layers is trained on the intents stored in data.json.
The neural network uses ReLU activation and is trained using SGD with momentum and Nesterov acceleration for better convergence.
After training, the model is saved and loaded into the Flask application for real-time classification of user inputs.
File Structure

app.py: Flask application to handle user interactions.
training.py: Script for training the chatbot's machine learning model.
data.json: Contains predefined intents, patterns, and responses.
model.h5: The trained model file.
texts.pkl & labels.pkl: Pickled files storing the processed words and labels for model training.
Future Enhancements

Adding more intents and responses to increase the chatbot's knowledge base.
Implementing more advanced NLP techniques like word embeddings.
Deploying the chatbot on cloud platforms such as AWS or Heroku.

License

This project is licensed under the MIT License. See the LICENSE file for details.

# Movie Interest Prediction App

A simple web application that predicts a user's movie interest category (Animation, Action, or Drama) based on their age and gender, using a **Decision Tree Classifier**.

---

## Features

- Predicts movie interests based on user input (age and gender)  
- Clean and intuitive web interface  
- Built using Python and Flask  
- Uses a Decision Tree Classifier for predictions  

---

## Prerequisites

Make sure the following are installed:

- Python 3.6 or higher  
- pip (Python package manager)

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/praveenasiva/movie_interest_app.git
cd movie_interest_app
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train the model (optional)

A pre-trained model (`model.pkl`) is already included. If you want to retrain the model:

```
python train_model.py
```

---

## Running the Application

Start the Flask development server:

```
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

---

## How It Works

- The app uses a **Decision Tree Classifier** trained on sample data from `Movie_Interests.csv`.  
- When a user submits their age and gender:
- The model predicts a movie interest category.
- The app displays the predicted interest (Animation, Action, or Drama).

---

## File Structure

```
movie_interest_app/
├── app.py                # Flask application
├── train_model.py        # Script to train the ML model
├── model.pkl             # Pre-trained model
├── interest_map.pkl      # Mapping of interest categories
├── Movie_Interests.csv   # Sample training data
├── requirements.txt      # Python dependencies
├── render.yaml           # Deployment configuration
└── README.md             # Project documentation
```

---

## Future Improvements

- Add more input features like location, favorite genre, etc.  
- Experiment with different machine learning models  
- Improve the UI using frontend frameworks



## Step-by-Step Guide: How to Use the Movie Interest Predictor
Step 1: Input Form
<img width="1914" height="1005" alt="Step-1" src="https://github.com/user-attachments/assets/f271a4ae-dd3b-4c15-8574-bac69b962d68" />

Step 2: Filled Form Example
<img width="1887" height="1009" alt="Step-2" src="https://github.com/user-attachments/assets/e1362bb2-aa82-47ef-99d3-914ded2e2d3f" />

Step 3: Prediction Result
<img width="1919" height="992" alt="Step-3" src="https://github.com/user-attachments/assets/436e2798-5de6-40bd-815f-b18df68d5226" />





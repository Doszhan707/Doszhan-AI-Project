# 🧠 AI Career Advisor

AI Career Advisor is a web-based intelligent system designed to recommend career paths based on user input such as education, interests, skills, work preference, and personality type. It uses **12 AI algorithms** (supervised, unsupervised, and computer vision) to analyze and predict the best-fit career options for the user.

## 🌐 Live Demo
> *Coming soon...*

---

## 🚀 Features

- ✅ Predicts career paths using **8 supervised learning algorithms**:
  - Linear Regression
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - KNN
  - Naive Bayes
  - SVM
  - Gradient Boosting

- ✅ Visualizes data using:
  - **KMeans clustering**
  - **PCA (Principal Component Analysis)**

- ✅ Detects frequent combinations using:
  - **FP-Growth (Association Rules)**

- ✅ Performs image analysis using:
  - **SIFT (Computer Vision)**

- ✅ Stylish interface in **mint-white color theme**
- ✅ Displays predictions, accuracy scores, clusters, PCA, association results, and CV results in a clean layout

---

## 🧩 Technologies Used

- **Python 3**
- **Flask** (backend)
- **HTML + CSS** (frontend)
- **NumPy**, **Pandas**, **Scikit-learn**
- **MLxtend**, **OpenCV**
- **Bootstrap-inspired CSS styling**

---

## 📂 Project Structure

```
AI_Career_Advisor/
│
├── ai_modules/
│   └── algorithms.py           # All 12 AI algorithms
│
├── static/
│   └── style.css               # Custom mint-white design
│   └── images/sample.png       # For CV testing
│
├── templates/
│   └── index.html              # Main interface
│
├── app.py                      # Flask server
├── requirements.txt            # Dependencies
└── README.md                   # Project overview
```

---

## ▶️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI_Career_Advisor.git
   cd AI_Career_Advisor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Visit `http://127.0.0.1:5000` in your browser.

---

## 📌 Sample Input Fields

- Education Level: High School / Bachelor / Master / PhD  
- Interests: Math, Biology, Programming, Law, Design (multi-select)  
- Skills: Python, Creativity, Communication, etc. (multi-select)  
- Work Preference: Remote / Office / Flexible  
- Personality: Analytical / Creative / Practical / Empathetic  

---

## 📬 License

This project is developed for educational purposes as part of a university AI final project. All assets are free to use under the MIT License.

---

## 📅 Author

Created with 💡 by Doszhan Erasy @ IITU Spring 2025

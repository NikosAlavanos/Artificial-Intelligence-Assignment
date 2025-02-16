# **AI Assignments Repository** 🤖📚

## **Overview**

This repository contains two major assignments for the Artificial Intelligence course (2024–25) at the University of Economics, Athens, Department of Informatics. The assignments showcase implementations of classical AI techniques and algorithms in different programming languages:

- **Assignment 1:**  
  - **Othello Game:** A Java implementation of the classic board game Reversi (Othello) where the computer opponent uses the MiniMax algorithm with alpha-beta pruning.  
  - **Forward Chaining for Horn Clauses:** An implementation in Java for performing forward chaining in propositional logic (specifically for Horn clauses), demonstrating inference techniques.

- **Assignment 2:**  
  - **Machine Learning Algorithms:** Python implementations of three classification algorithms applied to text data:
    - Naive Bayes (Bernoulli or Multinomial variants)
    - Random Forest (using ID3-based decision trees with a maximum depth hyperparameter)
    - AdaBoost (using decision stumps)
  
  These algorithms are applied to a sentiment analysis task on the IMDB Large Movie Review Dataset.

## **Key Features**

- **Assignment 1 (Java):**
  - **Othello Game:**
    - Implements classic game rules with an AI opponent powered by MiniMax and alpha-beta pruning.
    - Allows the user to select the search depth and choose whether to play first.
    - Displays the board state after every move.
  - **Forward Chaining for Horn Clauses:**
    - Reads a knowledge base from a text file and determines the truth of a given proposition.
    - Supports inference over definite clauses in propositional logic.
    - Provides optional outputs such as a proof tree and variable assignments.

- **Assignment 2 (Python):**
  - **Naive Bayes Classifier:**  
    Implements a text classification model using either Bernoulli or Multinomial forms.
  - **Random Forest Classifier:**  
    Constructs decision trees using an ID3-based approach with a controlled maximum depth.
  - **AdaBoost Algorithm:**  
    Uses decision stumps (depth-1 trees) to build a boosted ensemble for classification.
  - **Experimentation and Evaluation:**  
    - Uses feature selection based on information gain from a processed vocabulary.
    - Provides learning curves and evaluation metrics (precision, recall, F1) for training, development, and test sets.
    - Compares custom implementations with available libraries such as Scikit-learn.

## **How to Use**

### **Assignment 1 (Java)**
1. **Setup:**
   - Ensure you have Java 17 or later installed.
   - Import the project into your preferred Java IDE.
   - Place all necessary assets and configuration files in their designated directories.
2. **Running the Othello Game:**
   - Compile and run the Othello game module.
   - Follow on-screen instructions to choose the search depth and decide who plays first.
3. **Executing Forward Chaining:**
   - Run the forward chaining module.
   - Provide the knowledge base file and the target proposition either via command line or a simple GUI.
   - View the inference result and any additional information (e.g., proof tree).

### **Assignment 2 (Python)**
1. **Setup:**
   - Ensure Python 3 is installed along with the required libraries:
     - **numpy, matplotlib, seaborn, scikit‑learn, keras, torch (PyTorch), gensim, tensorflow**
   - Install dependencies.
   - Place the IMDB dataset (or a subset) in the designated data folder.
2. **Running the Experiments:**
   - Execute the provided Python scripts for each algorithm (naive_bayes.py, random_forest.py, adaboost.py).
   - Adjust hyperparameters as needed based on your experiments (e.g., vocabulary size, tree depth, regularization terms).
   - The scripts will generate learning curves and evaluation tables.
3. **Comparison:**
   - Optionally run scripts that compare your custom implementations with standard implementations from Scikit-learn.
   - Analyze the differences in performance metrics.

## **Technologies Used**

- **Java:**  
  For developing the Othello game and inference system (forward chaining for Horn clauses).  
  - **Java Swing:** Used for building the graphical user interface (Java 17+).

- **Python:**  
  For implementing and evaluating the machine learning algorithms.

- **Python Libraries:**  
  - **numpy**
  - **matplotlib**
  - **seaborn**
  - **scikit‑learn**
  - **keras**
  - **torch (PyTorch)**
  - **gensim**
  - **tensorflow**


## **Contributors**

* [Νίκος Αλαβάνος](https://github.com/NikosAlavanos)  
* [Κωσταντίνος Γιοβανόπουλος](https://github.com/Giovas2126)
* [Βασίλης Νικηφοράκης](https://github.com/NikiforakisV)

## **More Info**

For detailed documentation, experimental results, and further insights into each assignment, please refer to the accompanying PDF reports in each assignment folder.

---

Explore the repository to dive into our implementations of classic AI techniques and machine learning algorithms. Contributions and feedback are welcome!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import random

# Load the SpamAssassin dataset
# Replace 'path/to/spam/ham' with the actual path to your dataset
spam_df = pd.read_csv('path/to/spam/ham/spam_assassin.csv', encoding='latin-1')

# Drop missing values
spam_df = spam_df.dropna()

# Select relevant columns
spam_df = spam_df[['v1', 'v2']]

# Rename columns for clarity
spam_df.columns = ['label', 'text']

# Convert labels to numerical values
le = LabelEncoder()
spam_df['label'] = le.fit_transform(spam_df['label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spam_df['text'], spam_df['label'], test_size=0.2, random_state=42)

# Create a genetic algorithm for feature selection
def genetic_algorithm(X_train, y_train, generations=10, population_size=20):
    num_features = X_train.shape[1]
    population = []

    for _ in range(population_size):
        # Randomly select features for each individual
        features = np.random.choice(num_features, size=int(0.5 * num_features), replace=False)
        population.append(features)

    best_features = None
    best_fitness = 0

    for generation in range(generations):
        # Evaluate the fitness of each individual in the population
        for features in population:
            fitness = evaluate_fitness(features, X_train, y_train)
            
            # Update the best features if the current set has higher fitness
            if fitness > best_fitness:
                best_fitness = fitness
                best_features = features

        # Select parents and perform crossover to create a new generation
        population = create_next_generation(population)

    return best_features

# Function to evaluate the fitness of a feature set
def evaluate_fitness(features, X_train, y_train):
    clf = XGBClassifier()

    # Evaluate using cross-validation
    clf.fit(X_train.iloc[:, features], y_train)
    y_pred = clf.predict(X_train.iloc[:, features])

    # Fitness is the accuracy on the training set
    fitness = accuracy_score(y_train, y_pred)
    return fitness

# Function to create the next generation using crossover
def create_next_generation(population):
    next_generation = []

    for _ in range(len(population)):
        parent1 = random.choice(population)
        parent2 = random.choice(population)

        # Perform crossover
        crossover_point = random.randint(0, len(parent1))
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

        next_generation.append(child)

    return next_generation

# Main loop for the genetic algorithm
best_features = genetic_algorithm(X_train, y_train, generations=10, population_size=20)

# Use the best features to train the final XGBoost model
final_clf = XGBClassifier()

final_clf.fit(X_train.iloc[:, best_features], y_train)

# Evaluate on the test set
y_pred = final_clf.predict(X_test.iloc[:, best_features])
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy}")

import pandas as pd
import sqlite3
from scripts.predict_race import main  as predict_course

def connect_to_db(db_path):
    """Connect to the SQLite database."""
    return sqlite3.connect('data/hippique.db')

def fetch_courses(conn):
    """Fetch course data from the database."""
    query = ("SELECT comp, arriv FROM Courses_test LIMIT 0,30")

    return pd.read_sql_query(query, conn)

def execute_predictions(course_comp):
    """Simulate the execution of predictions for a given course."""
    # Remplacez ceci par votre logique de prédiction
    # Pour l'exemple, nous allons simplement retourner une liste de numéros prédits
    prediction = predict_course(course_comp)
    return prediction

def calculate_success_rates(predictions, actual_results):
    """Calculate success rates for the predictions."""
    # Convert predictions from string to list
    predicted_list = predictions.split('-')  # Convert '4-3-6-7-8' to ['4', '3', '6', '7', '8']

    # Extract the top 5 actual results from the actual_results string
    actual_list = actual_results.split('-')[:3]  # Get the first 5 results

    # Convert to sets for comparison
    predicted_set = set(predicted_list)
    actual_set = set(actual_list)

    # Check if the quinté is in order
    exact_match = predicted_list == actual_list
    # Check if the quinté is in disorder
    unordered_match = predicted_set == actual_set

    # Calculate the rate of correct numbers identified
    correct_numbers = predicted_set.intersection(actual_set)
    correct_rate = len(correct_numbers) / len(actual_set) * 100 if actual_set else 0

    return exact_match, unordered_match, correct_rate

def main():
    # Connexion à la base de données
    conn = connect_to_db('data/hippique.db')

    # Récupération des données des courses
    courses = fetch_courses(conn)
    conn.close()

    # Initialisation des résultats
    results = []

    for index, row in courses.iterrows():
        course_comp = row['comp']
        actual_results = row['arriv']  # Supposons que les résultats soient stockés sous forme de chaîne

        # Exécuter les prédictions
        predictions = execute_predictions(course_comp)

        # Calculer les taux de réussite
        exact_match, unordered_match, correct_rate = calculate_success_rates(predictions, actual_results)

        # Stocker les résultats
        results.append({
            'course_comp': course_comp,
            'predictions': predictions,
            'actual_results': actual_results,
            'exact_match': exact_match,
            'unordered_match': unordered_match,
            'correct_rate': correct_rate
        })

    # Convertir les résultats en DataFrame pour affichage
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

if __name__ == "__main__":
    main()
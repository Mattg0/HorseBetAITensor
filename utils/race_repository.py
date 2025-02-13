from typing import List, Dict, Optional
import sqlite3
from datetime import datetime
import json


class RaceRepository:
    def __init__(self, db):
        self.db = db

    def store_race(self, race_data: Dict):
        """Store a race in the database."""
        with self.db.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO daily_races (
                    comp, jour, hippodrome, reun, prix, heure, prixnom,
                    meteo, dist, corde, natpis, pistegp, typec,
                    temperature, forceVent, directionVent, nebulosite,
                    participants
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                race_data['comp'],
                race_data['jour'],
                race_data['hippodrome'],
                race_data['reun'],
                race_data['prix'],
                race_data['heure'],
                race_data['prixnom'],
                race_data['meteo'],
                race_data['dist'],
                race_data['corde'],
                race_data['natpis'],
                race_data['pistegp'],
                race_data['typec'],
                race_data['temperature'],
                race_data['forceVent'],
                race_data['directionVent'],
                race_data['nebulosite'],
                race_data['participants']
            ))

    def get_unpredicted_races(self, date: str) -> List[Dict]:
        """Get races that haven't been predicted yet."""
        with self.db.get_connection() as conn:
            cursor = conn.execute('''
                SELECT *
                FROM daily_races
                WHERE jour = ? AND predicted = 0
            ''', (date,))

            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def mark_race_predicted(self, comp: int):
        """Mark a race as predicted."""
        with self.db.get_connection() as conn:
            conn.execute('''
                UPDATE daily_races 
                SET predicted = 1
                WHERE comp = ?
            ''', (comp,))

    def store_prediction(self, comp: int, bet_type: str, sequence: str, confidence: float):
        """Store a prediction result."""
        with self.db.get_connection() as conn:
            conn.execute('''
                INSERT INTO predictions (comp, bet_type, sequence, confidence)
                VALUES (?, ?, ?, ?)
            ''', (comp, bet_type, sequence, confidence))

    def migrate_to_training_db(self, training_db_path: str):
        """Migrate predicted races to training database."""
        with self.db.get_connection() as temp_conn:
            with sqlite3.connect(training_db_path) as train_conn:
                # Get all predicted races
                races = temp_conn.execute('''
                    SELECT jour, comp, hippodrome, meteo, dist, corde, 
                           natpis, pistegp, typec, temperature, forceVent, 
                           directionVent, nebulosite, participants
                    FROM daily_races
                    WHERE predicted = 1
                ''').fetchall()

                # Insert into training database
                train_conn.executemany('''
                    INSERT INTO Course (
                        jour, comp, hippodrome, meteo, dist, corde,
                        natpis, pistegp, typec, temperature, forceVent,
                        directionVent, nebulosite, participants
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', races)
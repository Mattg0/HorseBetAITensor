from pathlib import Path
import sqlite3
from typing import List, Dict, Optional
import json
import pandas as pd
from datetime import datetime

from env_setup import setup_environment, get_database_path
from core.database import Database


class DailyRaceToHistorical:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the migration handler with configuration."""
        self.config = setup_environment(config_path)
        self.db = Database(config_path)
        self.historical_db_path = get_database_path(self.config)

    def _get_race_data(self, comp_id: int, conn: sqlite3.Connection) -> Optional[Dict]:
        """
        Get all necessary data for a race from daily_races and predictions.
        """
        try:
            # Get race details
            cursor = conn.execute("""
                SELECT dr.*, p.sequence as predicted_sequence, p.confidence,
                       r.ordre_arrivee, r.created_at as result_created
                FROM daily_races dr
                LEFT JOIN predictions p ON dr.comp = p.comp
                LEFT JOIN Resultats r ON dr.comp = r.comp
                WHERE dr.comp = ?
            """, (comp_id,))

            columns = [description[0] for description in cursor.description]
            race_data = cursor.fetchone()

            if not race_data:
                return None

            return dict(zip(columns, race_data))

        except sqlite3.Error as e:
            print(f"Error fetching race data for comp_id {comp_id}: {e}")
            return None

    def migrate_race(self, comp_id: int) -> bool:
        """
        Migrate a single race from daily_races to historical database.
        """
        print(f"\nMigrating race {comp_id} to historical database...")

        try:
            # Get connections to both databases
            daily_conn = self.db.get_connection()
            historical_conn = sqlite3.connect(self.historical_db_path)

            # Get race data
            race_data = self._get_race_data(comp_id, daily_conn)
            if not race_data:
                print(f"No data found for race {comp_id}")
                return False

            # Begin transaction
            historical_conn.execute("BEGIN")

            try:
                # Insert into Course table
                historical_conn.execute("""
                    INSERT INTO Course (
                        comp, jour, hippodrome, meteo, dist, corde,
                        natpis, pistegp, typec, temperature, forceVent,
                        directionVent, nebulosite, participants
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_data['comp'],
                    race_data['jour'],
                    race_data['hippodrome'],
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

                # Insert prediction if exists
                if race_data.get('predicted_sequence'):
                    historical_conn.execute("""
                        INSERT INTO Predictions (
                            comp, sequence, confidence, created_at
                        ) VALUES (?, ?, ?, datetime('now'))
                    """, (
                        race_data['comp'],
                        race_data['predicted_sequence'],
                        race_data['confidence']
                    ))

                # Insert results if exists
                if race_data.get('ordre_arrivee'):
                    historical_conn.execute("""
                        INSERT INTO Resultats (
                            comp, ordre_arrivee, created_at
                        ) VALUES (?, ?, ?)
                    """, (
                        race_data['comp'],
                        race_data['ordre_arrivee'],
                        race_data['result_created']
                    ))

                # Commit transaction
                historical_conn.commit()
                print(f"Successfully migrated race {comp_id}")
                return True

            except Exception as e:
                historical_conn.rollback()
                print(f"Error during migration of race {comp_id}: {e}")
                return False

        except Exception as e:
            print(f"Error connecting to databases: {e}")
            return False
        finally:
            daily_conn.close()
            historical_conn.close()

    def migrate_races(self, comp_ids: List[int]) -> bool:
        """
        Migrate multiple races from daily_races to historical database.
        """
        print(f"\nMigrating {len(comp_ids)} races to historical database...")

        success_count = 0
        for comp_id in comp_ids:
            if self.migrate_race(comp_id):
                success_count += 1
            else:
                print(f"Failed to migrate race {comp_id}")

        print(f"\nMigration completed: {success_count}/{len(comp_ids)} races successfully migrated")
        return success_count == len(comp_ids)

    def verify_migration(self, comp_ids: List[int]) -> bool:
        """
        Verify that races were correctly migrated to historical database.
        """
        print("\nVerifying migration...")

        try:
            historical_conn = sqlite3.connect(self.historical_db_path)

            for comp_id in comp_ids:
                # Check Course table
                cursor = historical_conn.execute(
                    "SELECT COUNT(*) FROM Course WHERE comp = ?",
                    (comp_id,)
                )
                if cursor.fetchone()[0] == 0:
                    print(f"Race {comp_id} not found in Course table")
                    return False

                # Check Predictions table
                cursor = historical_conn.execute(
                    "SELECT COUNT(*) FROM Predictions WHERE comp = ?",
                    (comp_id,)
                )
                if cursor.fetchone()[0] == 0:
                    print(f"Warning: No predictions found for race {comp_id}")

                # Check Resultats table
                cursor = historical_conn.execute(
                    "SELECT COUNT(*) FROM Resultats WHERE comp = ?",
                    (comp_id,)
                )
                if cursor.fetchone()[0] == 0:
                    print(f"Warning: No results found for race {comp_id}")

            return True

        except Exception as e:
            print(f"Error verifying migration: {e}")
            return False
        finally:
            historical_conn.close()
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import List, Dict
import sys

from env_setup import setup_environment


class HistoryMigrator:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize history migrator."""
        self.config = setup_environment(config_path)
        self.source_db = self._get_db_connection('full')

        # Get path for historical database
        db_config = next((db for db in self.config['databases'] if db['name'] == 'full'), None)
        if not db_config:
            raise ValueError("Historical database configuration not found")
        self.hist_db_path = Path(self.config['rootdir']) / db_config['path']

    def _get_db_connection(self, db_name: str) -> sqlite3.Connection:
        """Get database connection."""
        db_config = next((db for db in self.config['databases'] if db['name'] == db_name), None)
        if not db_config:
            raise ValueError(f"Database '{db_name}' not found in configuration")
        db_path = Path(self.config['rootdir']) / db_config['path']
        return sqlite3.connect(db_path)

    def get_completed_races(self) -> List[Dict]:
        """Get completed races that need to be migrated."""
        today = datetime.now().strftime("%Y-%m-%d")
        cursor = self.source_db.execute("""
            SELECT dr.*, r.ordre_arrivee
            FROM daily_races dr
            JOIN Resultats r ON dr.comp = r.comp
            WHERE dr.jour = ?
            AND NOT EXISTS (
                SELECT 1 FROM Course c WHERE c.comp = dr.comp
            )
        """, (today,))

        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def migrate_race(self, race_data: Dict) -> bool:
        """Migrate a single race to historical database."""
        try:
            # Insert into Course table
            self.source_db.execute("""
                INSERT INTO Course (
                    comp, jour, hippodrome, reun, prix, heure, prixnom,
                    meteo, dist, corde, natpis, pistegp, typec,
                    temperature, forceVent, directionVent, nebulosite
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_data['comp'], race_data['jour'], race_data['hippodrome'],
                race_data['reun'], race_data['prix'], race_data['heure'],
                race_data['prixnom'], race_data['meteo'], race_data['dist'],
                race_data['corde'], race_data['natpis'], race_data['pistegp'],
                race_data['typec'], race_data['temperature'], race_data['forceVent'],
                race_data['directionVent'], race_data['nebulosite']
            ))

            # Parse participants and insert into Partants table
            participants = json.loads(race_data['participants'])
            for participant in participants:
                self.source_db.execute("""
                    INSERT INTO Partants (
                        comp, idche, cheval, numero, age, musiqueche,
                        idJockey, musiquejoc, idEntraineur, cotedirect
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_data['comp'], participant['idche'], participant['cheval'],
                    participant['numero'], participant['age'], participant['musiqueche'],
                    participant['idJockey'], participant['musiquejoc'],
                    participant['idEntraineur'], participant['cotedirect']
                ))

            # Copy results
            self.source_db.execute("""
                INSERT INTO Resultats (comp, ordre_arrivee)
                SELECT comp, ordre_arrivee
                FROM Resultats
                WHERE comp = ?
            """, (race_data['comp'],))

            self.source_db.commit()
            return True

        except Exception as e:
            print(f"Error migrating race {race_data['comp']}: {e}")
            self.source_db.rollback()
            return False

    def cleanup_migrated_races(self, comp_ids: List[int]):
        """Clean up migrated races from daily_races table."""
        if not comp_ids:
            return

        placeholders = ','.join('?' * len(comp_ids))
        self.source_db.execute(f"""
            DELETE FROM daily_races
            WHERE comp IN ({placeholders})
        """, comp_ids)
        self.source_db.commit()

    def run(self):
        """Run the migration process."""
        print("Starting historical data migration...")
        races = self.get_completed_races()

        if not races:
            print("No completed races to migrate")
            return

        migrated_comp_ids = []
        for race in races:
            try:
                print(f"\nMigrating race {race['comp']}...")
                if self.migrate_race(race):
                    migrated_comp_ids.append(race['comp'])
                    print(f"Successfully migrated race {race['comp']}")
                else:
                    print(f"Failed to migrate race {race['comp']}")
            except Exception as e:
                print(f"Error processing race {race['comp']}: {str(e)}")
                continue

        if migrated_comp_ids:
            print("\nCleaning up migrated races...")
            self.cleanup_migrated_races(migrated_comp_ids)

        print("\nMigration completed!")


def main():
    migrator = HistoryMigrator()
    migrator.run()


if __name__ == "__main__":
    main()
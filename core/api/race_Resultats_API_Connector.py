import requests
import sqlite3
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional
import time

from env_setup import setup_environment,get_database_path


class ResultsFetcher:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize results fetcher."""
        self.config = setup_environment(config_path)
        self.api_key = self.config['aspiturf']['api_key']
        self.base_url = "https://api.aspiturf.com/api"
        self.db = self._get_db_connection()

    def _get_db_connection(self) -> sqlite3.Connection:
        db_path = get_database_path(self.config)
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA cache_size = -2000')
        conn.execute('PRAGMA synchronous = NORMAL')
        return conn

    def get_todays_races(self) -> List[Dict]:
        """Get all races for today that need results."""
    #    today = datetime.now().strftime("%Y-%m-%d")
        today='2025-02-13'
        cursor = self.db.execute("""
            SELECT dr.comp, dr.reun, dr.prix
            FROM daily_races dr
            LEFT JOIN Resultats r ON dr.comp = r.comp
            WHERE dr.jour = ? AND r.comp IS NULL
            ORDER BY dr.heure
        """, (today,))
        return [{'comp': row[0], 'reun': row[1], 'prix': row[2]} for row in cursor.fetchall()]

    def fetch_race_results(self, date: str, reun: str, prix: int) -> Optional[List[Dict]]:
        """Fetch results for a specific race from API."""
        params = {
            "uid": self.api_key,
            "jour[]": date,
            "r": reun,
            "c": prix
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Add debug logging
            print(f"\nFetched data for R{reun}C{prix}:")
            if data:
                first_horse = data[0] if isinstance(data, list) and len(data) > 0 else None
                if first_horse:
                    print(f"Sample horse data:")
                    print(f"  CL value: {first_horse.get('cl', 'Not found')}")
                    print(f"  Number: {first_horse.get('numero', 'Not found')}")
                    print(f"  Horse ID: {first_horse.get('idChe', 'Not found')}")
                    print(f"  Horse Name: {first_horse.get('cheval', 'Not found')}")

            return data

        except requests.RequestException as e:
            print(f"Error fetching results: {e}")
            return None

    def store_results(self, comp: int, results_data: List[Dict]):
        try:
            # Extract ordre_arrivee (finishing order)
            ordre_arrivee = []

            for horse in results_data:
                # Get cl directly from horse object (not in numcourse)
                cl = horse.get('cl')
                if cl is None:
                    continue
                try:
                    # Ensure narrivee is converted to integer
                    narrivee = int(cl) if cl.isdigit() else 99
                except (ValueError, AttributeError):
                    narrivee = 99

                ordre_arrivee.append({
                    'numero': str(horse.get('numero')),  # Ensure numero is string
                    'idche': horse.get('idChe'),
                    'narrivee': narrivee  # This should now always be an integer
                })

            if not ordre_arrivee:
                print(f"No valid finish positions found for race {comp}")
                return False

            # Sort by finishing position (lower numbers first, 99s at the end)
            ordre_arrivee.sort(key=lambda x: x['narrivee'])

            # Store in database
            self.db.execute("""
                INSERT INTO Resultats (comp, ordre_arrivee, created_at)
                VALUES (?, ?, datetime('now'))
            """, (comp, json.dumps(ordre_arrivee)))
            self.db.commit()

            return True
        except Exception as e:
            print(f"Error storing results for race {comp}: {e}")
            return False
        except Exception as e:
            print(f"Error storing results for race {comp}: {e}")
            return False

    def run(self):
        """Fetch and store results for all of today's races."""
        print("Starting results collection...")
        today = datetime.now().strftime("%Y-%m-%d")
        #today = '2025-02-13'
        races = self.get_todays_races()

        for race in races:
            try:
                print(f"\nFetching results for race {race['comp']}")
                results = self.fetch_race_results(today, race['reun'], race['prix'])

                if results:
                    success = self.store_results(race['comp'], results)
                    if success:
                        print(f"Stored results for race {race['comp']}")
                    else:
                        print(f"Failed to store results for race {race['comp']}")

                # Add small delay between API calls
                time.sleep(1)

            except Exception as e:
                print(f"Error processing race {race['comp']}: {str(e)}")
                continue




def main():
    fetcher = ResultsFetcher()
    fetcher.run()


if __name__ == "__main__":
    main()
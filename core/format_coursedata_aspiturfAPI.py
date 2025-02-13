import requests
import json
from decimal import Decimal
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime


class RaceDataFetcher:
    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.base_url = "https://api.aspiturf.com/api"

    def fetch_daily_races(self, date: str) -> List[Dict]:
        """Fetch list of all races for a given date."""
        params = {
            "uid": self.api_key,
            "jour[]": date
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract unique race identifiers
            races = []
            seen = set()

            for entry in data:
                numcourse = entry.get('numcourse', {})
                reun = numcourse.get('reun')
                prix = numcourse.get('prix')
                comp = numcourse.get('comp')

                if reun and prix and comp:
                    identifier = (reun, prix)
                    if identifier not in seen:
                        seen.add(identifier)
                        races.append({
                            'reun': reun,
                            'prix': prix,
                            'comp': comp,
                            'hippo': numcourse.get('hippo'),
                            'heure': numcourse.get('heure'),
                            'prixnom': numcourse.get('prixnom')
                        })

            return races

        except requests.RequestException as e:
            print(f"Error fetching daily races: {e}")
            return []

    def fetch_race_details(self, date: str, reun: str, prix: int) -> Optional[List[Dict]]:
        """Fetch detailed data for a specific race."""
        params = {
            "uid": self.api_key,
            "jour[]": date,
            "r": reun,
            "c": prix
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching race details (R{reun}C{prix}): {e}")
            return None

    def safe_convert_to_float(self, value: Any) -> float:
        """Safely convert any value to float."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Decimal):
            return float(value)
        if not value:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def format_race_data(self, race_data: List[Dict]) -> Optional[Dict]:
        """Format race data to match format_coursedata.py output."""
        if not race_data or not race_data[0].get('numcourse'):
            return None

        first_entry = race_data[0]
        numcourse = first_entry['numcourse']

        # Extract course info following the same structure as format_coursedata
        course_info = {
            'comp': numcourse.get('comp'),
            'hippo': numcourse.get('hippo'),
            'jour': numcourse.get('jour'),
            'meteo': numcourse.get('meteo'),
            'dist': self.safe_convert_to_float(numcourse.get('dist')),
            'corde': numcourse.get('corde'),
            'natpis': numcourse.get('natpis'),
            'pistegp': numcourse.get('pistegp'),
            'typec': numcourse.get('typec'),
            'temperature': self.safe_convert_to_float(numcourse.get('temperature')),
            'forceVent': self.safe_convert_to_float(numcourse.get('forceVent')),
            'directionVent': numcourse.get('directionVent'),
            'nebulositeLibelleCourt': numcourse.get('nebulositeLibelleCourt')
        }

        # Extract participants data following the same structure as format_coursedata
        participants = []
        for entry in race_data:
            participant = {
                'idche': self.safe_convert_to_float(entry.get('idChe')),
                'cheval': entry.get('cheval'),
                'numero': int(entry.get('numero', 0)),
                'age': int(entry.get('age', 0)),
                'musiqueche': entry.get('musiqueche', ''),
                'idJockey': self.safe_convert_to_float(entry.get('idJockey')),
                'musiquejoc': entry.get('musiquejoc', ''),
                'idEntraineur': self.safe_convert_to_float(entry.get('idEntraineur')),
                'cotedirect': self.safe_convert_to_float(entry.get('cotedirect'))
            }
            participants.append(participant)

        return {
            'course_info': course_info,
            'participants': participants
        }


def main(comp_id: int = None) -> Optional[str]:
    """Main function to fetch and transform race data."""
    api_key = "8cdfGeF4pHeSOPv05dPnVyGaghL2"
    fetcher = RaceDataFetcher(api_key)

    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")

    # First, get the list of all races for today
    races = fetcher.fetch_daily_races(today)

    if not races:
        print("No races found for today")
        return None

    # If comp_id is provided, find and fetch the specific race
    if comp_id:
        target_race = next((r for r in races if r['comp'] == comp_id), None)
        if not target_race:
            print(f"No race found with comp_id {comp_id}")
            return None

        # Fetch and format the specific race
        race_data = fetcher.fetch_race_details(today, target_race['reun'], target_race['prix'])
        if race_data:
            formatted_race = fetcher.format_race_data(race_data)
            if formatted_race:
                return json.dumps(formatted_race)
    else:
        # Print available races
        print("\nAvailable races today:")
        for race in races:
            print(
                f"Comp ID: {race['comp']}, {race['hippo']} R{race['reun']}C{race['prix']} - {race['heure']} - {race['prixnom']}")

    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        try:
            comp_id = int(sys.argv[1])
            result = main(comp_id)
            if result:
                print(result)
        except ValueError:
            print("Error: Competition ID must be a number")
    else:
        main()  # Print list of available races
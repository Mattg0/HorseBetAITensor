import requests
from typing import List, Dict, Optional
from datetime import datetime


class RaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.aspiturf.com/api"

    def fetch_daily_races(self, date: str) -> List[Dict]:
        """Fetch list of races for a given date."""
        params = {
            "uid": self.api_key,
            "jour[]": date
        }

        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_race_details(self, date: str, reun: str, prix: int) -> Optional[Dict]:
        """Fetch detailed data for a specific race."""
        params = {
            "uid": self.api_key,
            "jour[]": date,
            "r": reun,
            "c": prix
        }

        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()
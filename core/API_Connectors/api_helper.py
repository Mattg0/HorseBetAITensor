from core.DB_Connectors import requests


def fetch_participants(url):
    """Fetch participants data from the API."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None
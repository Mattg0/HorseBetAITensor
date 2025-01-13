import json
from core.db_helper  import create_connection, insert_participant
from core.api_helper import fetch_participants

def main():
    url = "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/12102019/R1/C1/participants"

    # Fetch participants data
    participants_data = fetch_participants(url)
    if participants_data is None:
        return

    # Connect to the database
    database = "data/hippique.db"
    conn = create_connection(database)

    # Insert participants into the database
    for participant in participants_data['participants']:
        insert_participant(conn, participant)

    conn.close()
    print("Participants inserted successfully.")

if __name__ == "__main__":
    main()
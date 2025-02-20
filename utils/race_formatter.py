from typing import Dict, List, Optional
from decimal import Decimal
import json


class RaceFormatter:
    @staticmethod
    def format_race_data(race_data: List[Dict]) -> Optional[Dict]:
        """
        Format race data for storage, including both race info and results.
        Returns a dict with 'race' and 'ordre_arrivee' keys.
        """
        try:
            if not race_data or not len(race_data):
                return None

            # Get race info from first entry
            first_entry = race_data[0]
            numcourse = first_entry['numcourse']
            comp_id = numcourse['comp']

            # Format race data
            formatted_race = {
                'comp': comp_id,
                'jour': numcourse.get('jour'),
                'hippodrome': numcourse.get('hippo'),
                'meteo': numcourse.get('meteo', ''),
                'dist': numcourse.get('dist', 0),
                'corde': numcourse.get('corde', ''),
                'natpis': numcourse.get('natpis', ''),
                'pistegp': numcourse.get('pistegp', ''),
                'typec': numcourse.get('typec', ''),
                'temperature': numcourse.get('temperature', 0),
                'forceVent': numcourse.get('forceVent', 0),
                'directionVent': numcourse.get('directionVent', ''),
                'nebulosite': numcourse.get('nebulosite', '')
            }

            # Format participants data
            participants = []
            ordre_arrivee = []

            for horse in race_data:
                # Format participant data
                participant = {
                    'idche': horse.get('idChe'),
                    'cheval': horse.get('cheval'),
                    'numero': horse.get('numero'),
                    'age': horse.get('age'),
                    'musiqueche': horse.get('musiqueche', ''),
                    'idJockey': horse.get('idJockey'),
                    'musiquejoc': horse.get('musiquejoc', ''),
                    'idEntraineur': horse.get('idEntraineur'),
                    'cotedirect': horse.get('cotedirect', 0)
                }
                participants.append(participant)

                # Extract finishing position for ordre_arrivee
                cl = horse.get('cl')
                if cl is not None:
                    try:
                        narrivee = int(cl) if cl.isdigit() else 99
                        ordre_arrivee.append({
                            'numero': str(horse.get('numero')),
                            'idche': horse.get('idChe'),
                            'narrivee': narrivee
                        })
                    except (ValueError, AttributeError):
                        continue

            # Sort ordre_arrivee by finishing position
            ordre_arrivee.sort(key=lambda x: x['narrivee'])

            # Add participants to race data
            formatted_race['participants'] = json.dumps(participants)

            return {
                'race': formatted_race,
                'ordre_arrivee': ordre_arrivee
            }

        except Exception as e:
            print(f"Error formatting race data: {e}")
            return None
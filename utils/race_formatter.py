from typing import Dict, List, Optional
from decimal import Decimal
import json


class RaceFormatter:
    @staticmethod
    def format_race_data(race_data: List[Dict]) -> Optional[Dict]:
        """Format race data to match Course table structure."""
        if not race_data or not race_data[0].get('numcourse'):
            return None

        def safe_float(value):
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, Decimal):
                return float(value)
            try:
                return float(value) if value not in ['N/A', '', None] else 0.0
            except (ValueError, TypeError):
                return 0.0

        first_entry = race_data[0]
        numcourse = first_entry['numcourse']

        # Format race info matching Course table structure
        race_info = {
            'comp': numcourse.get('comp'),
            'hippodrome': numcourse.get('hippo'),
            'jour': numcourse.get('jour'),
            'reun': numcourse.get('reun'),
            'prix': numcourse.get('prix'),
            'heure': numcourse.get('heure'),
            'prixnom': numcourse.get('prixnom'),
            'meteo': numcourse.get('meteo'),
            'dist': int(safe_float(numcourse.get('dist'))),
            'corde': numcourse.get('corde'),
            'natpis': numcourse.get('natpis'),
            'pistegp': numcourse.get('pistegp'),
            'typec': numcourse.get('typec'),
            'temperature': safe_float(numcourse.get('temperature')),
            'forceVent': safe_float(numcourse.get('forceVent')),
            'directionVent': numcourse.get('directionVent'),
            'nebulosite': numcourse.get('nebulositeLibelleCourt')
        }

        # Format participants as JSON string
        participants = []
        for entry in race_data:
            participant = {
                'idche': safe_float(entry.get('idChe')),
                'cheval': entry.get('cheval'),
                'numero': int(entry.get('numero', 0)),
                'age': int(entry.get('age', 0)),
                'musiqueche': entry.get('musiqueche', ''),
                'idJockey': safe_float(entry.get('idJockey')),
                'musiquejoc': entry.get('musiquejoc', ''),
                'idEntraineur': safe_float(entry.get('idEntraineur')),
                'cotedirect': safe_float(entry.get('cotedirect'))
            }
            participants.append(participant)

        race_info['participants'] = json.dumps(participants)
        return race_info
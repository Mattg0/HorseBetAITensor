import sqlite3


conn = sqlite3.connect('../../data/hippique.db')
cursor = conn.cursor()
a=cursor.execute('''
SELECT Count(*) From Course where cotedirecte"='' ;

       ''')

columns = cursor.description
result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cursor.fetchall()]

conn.close()
print(result)
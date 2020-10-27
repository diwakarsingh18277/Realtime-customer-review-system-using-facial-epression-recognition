import sqlite3
conn = sqlite3.connect('customers.sqlite')
cursor = conn.cursor()
print("Opened database successfully")

# _____________________ Create Table _____________________________________
cursor.execute(''' create table customer
        (Id BLOB primary key not null,
        Name text,
        Entry_time text,
        Entry_happiness real,
        Exit_happiness real,
        Exit_time text);''')
cursor.close()
# _______________ comment this secotion if table is already created _______

conn.commit()
conn.close()

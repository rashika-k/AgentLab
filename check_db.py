import sqlite3
con = sqlite3.connect("data/chinook.sqlite")
print(con.execute("SELECT Country, COUNT(*) FROM Customer GROUP BY Country ORDER BY 2 DESC LIMIT 3").fetchall())

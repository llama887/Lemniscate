import sqlite3

# Connect to your SQLite database (replace 'your_database.db' with the actual database file)
conn = sqlite3.connect("vector_database.db")
cursor = conn.cursor()

# Execute the query to retrieve all values from the conversations table
cursor.execute("SELECT * FROM conversations")

# Fetch all rows from the result of the query
rows = cursor.fetchall()

# Print each row
for row in rows:
    print(row)

# Close the database connection
conn.close()

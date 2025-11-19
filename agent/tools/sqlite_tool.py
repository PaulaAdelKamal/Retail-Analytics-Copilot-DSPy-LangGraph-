import sqlite3

def run_sqlite_query(query):
    """
    Connects to northwind.sqlite, executes the given SQL query, and returns
    the rows and column names.

    Args:
        query: The SQL query string to execute.

    Returns:
        A tuple containing the list of rows and a list of column names.
        If an error occurs, it returns a tuple with the error message string and None.
    """
    try:
        with sqlite3.connect('northwind.sqlite') as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            # Return an empty list for column names if the query doesn't produce columns (e.g., INSERT, UPDATE)
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            return rows, column_names
    except sqlite3.Error as e:
        return str(e), None

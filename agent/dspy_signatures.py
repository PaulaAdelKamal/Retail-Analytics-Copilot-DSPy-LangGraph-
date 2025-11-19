import dspy
from typing import Literal
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

class RouterSignature(dspy.Signature):
    """
    Classifies a question into 'SQL', 'RAG', or 'Hybrid'.
    """
    question: str = dspy.InputField(desc="The user's question.")
    classification: Literal['SQL', 'RAG', 'Hybrid']  = dspy.OutputField(
        desc="The classification of the question. Must be one of 'SQL', 'RAG', or 'Hybrid'."
    )

class TextToSQLSignature(dspy.Signature):
    """
    Converts a natural language question into a SQL query.
    """
    question = dspy.InputField(desc="The user's question.")
    schema = dspy.InputField(desc="The database schema.")
    sql_query = dspy.OutputField(desc="The generated SQL query.")

class SynthesizerSignature(dspy.Signature):
    """
    Synthesizes a final answer from the given context, providing citations.
    The output must be a single, valid JSON object.
    """
    question = dspy.InputField(desc="The user's question.")
    context = dspy.InputField(desc="Combined results from SQL queries and document chunks.")
    sql_query = dspy.InputField(desc="The last executed SQL query. Can be empty if no SQL was run.")
    format_hint = dspy.InputField(desc="A hint for how to format the output (e.g., 'list', 'table').")
    json_output = dspy.OutputField(
        desc="""A single, valid JSON object with the following schema:
{
    "id": "a unique identifier for this response",
    "final_answer": "The final answer, formatted according to the format_hint.",
    "sql": "The last executed SQL query, or an empty string if none was run.",
    "confidence": "A float from 0.0 to 1.0 representing confidence in the answer.",
    "explanation": "A brief, one to two-sentence explanation of how the answer was derived.",
    "citations": "A list of strings identifying the sources used (e.g., table names or document chunks like 'filename::chunk_id')."
}
"""
    )

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.route = dspy.Predict(RouterSignature)

    def forward(self, question):
        return self.route(question=question)

class TextToSQL(dspy.ChainOfThought):
    def __init__(self):
        super().__init__(TextToSQLSignature)

class Synthesizer(dspy.ChainOfThought):
    def __init__(self):
        super().__init__(SynthesizerSignature)
        
llm = dspy.LM("ollama_chat/phi3.5", api_base="http://localhost:11434")
dspy.settings.configure(lm=llm)

# --- 2. Create a Tiny Training Set ---
# These are examples of the input (question, schema) and the desired output (sql_query)
schema = """
CREATE TABLE [Customers](
    [CustomerID] TEXT,
    [CompanyName] TEXT,
    [ContactName] TEXT,
    [ContactTitle] TEXT,
    [Address] TEXT,
    [City] TEXT,
    [Region] TEXT,
    [PostalCode] TEXT,
    [Country] TEXT,
    [Phone] TEXT,
    [Fax] TEXT,
    PRIMARY KEY (`CustomerID`)
)
CREATE TABLE [Order Details](
   [OrderID]INTEGER NOT NULL,
   [ProductID]INTEGER NOT NULL,
   [UnitPrice]NUMERIC NOT NULL DEFAULT 0,
   [Quantity]INTEGER NOT NULL DEFAULT 1,
   [Discount]REAL NOT NULL DEFAULT 0,
    PRIMARY KEY ("OrderID","ProductID"),
    CHECK ([Discount]>=(0) AND [Discount]<=(1)),
    CHECK ([Quantity]>(0)),
    CHECK ([UnitPrice]>=(0)),
	FOREIGN KEY ([OrderID]) REFERENCES [Orders] ([OrderID]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
	FOREIGN KEY ([ProductID]) REFERENCES [Products] ([ProductID]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
)

CREATE TABLE [Orders](
   [OrderID]INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
   [CustomerID]TEXT,
   [EmployeeID]INTEGER,
   [OrderDate]DATETIME,
   [RequiredDate]DATETIME,
   [ShippedDate]DATETIME,
   [ShipVia]INTEGER,
   [Freight]NUMERIC DEFAULT 0,
   [ShipName]TEXT,
   [ShipAddress]TEXT,
   [ShipCity]TEXT,
   [ShipRegion]TEXT,
   [ShipPostalCode]TEXT,
   [ShipCountry]TEXT,
	FOREIGN KEY ([CustomerID]) REFERENCES [Customers] ([CustomerID]) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
)

CREATE TABLE [Products](
   [ProductID]INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
   [ProductName]TEXT NOT NULL,
   [SupplierID]INTEGER,
   [CategoryID]INTEGER,
   [QuantityPerUnit]TEXT,
   [UnitPrice]NUMERIC DEFAULT 0,
   [UnitsInStock]INTEGER DEFAULT 0,
   [UnitsOnOrder]INTEGER DEFAULT 0,
   [ReorderLevel]INTEGER DEFAULT 0,
   [Discontinued]TEXT NOT NULL DEFAULT '0',
    CHECK ([UnitPrice]>=(0)),
    CHECK ([ReorderLevel]>=(0)),
    CHECK ([UnitsInStock]>=(0)),
    CHECK ([UnitsOnOrder]>=(0)),

)
"""

trainset = [
    # --- Basic Selects & Filtering ---
    #dspy.Example(question="How many products are there?", schema=schema, sql_query="SELECT count(*) FROM Products").with_inputs("question", "schema"),
    #dspy.Example(question="What is the average price of all products?", schema=schema, sql_query="SELECT avg(UnitPrice) FROM Products").with_inputs("question", "schema"),
    #dspy.Example(question="List the names of products that are out of stock.", schema=schema, sql_query="SELECT ProductName FROM Products WHERE UnitsInStock = 0").with_inputs("question", "schema"),
    #dspy.Example(question="Find all customers located in London.", schema=schema, sql_query="SELECT * FROM Customers WHERE City = 'London'").with_inputs("question", "schema"),
    #dspy.Example(question="List the names of products that are discontinued.", schema=schema, sql_query="SELECT ProductName FROM Products WHERE Discontinued = '1'").with_inputs("question", "schema"),
    #dspy.Example(question="Which orders had a freight cost higher than 100?", schema=schema, sql_query="SELECT OrderID, Freight FROM Orders WHERE Freight > 100").with_inputs("question", "schema"),
    #dspy.Example(question="Show all products with a unit price between 10 and 20.", schema=schema, sql_query="SELECT ProductName, UnitPrice FROM Products WHERE UnitPrice >= 10 AND UnitPrice <= 20").with_inputs("question", "schema"),
    #dspy.Example(question="What are the contact names for customers in Germany?", schema=schema, sql_query="SELECT ContactName FROM Customers WHERE Country = 'Germany'").with_inputs("question", "schema"),
    #dspy.Example(question="List all customers who do not have a Fax number.", schema=schema, sql_query="SELECT CompanyName FROM Customers WHERE Fax IS NULL").with_inputs("question", "schema"),
    #dspy.Example(question="Which products need to be reordered? (Stock is less than reorder level)", schema=schema, sql_query="SELECT ProductName FROM Products WHERE UnitsInStock < ReorderLevel").with_inputs("question", "schema"),

    # --- Aggregations & Math ---
    #dspy.Example(question="What is the most expensive product?", schema=schema, sql_query="SELECT ProductName, UnitPrice FROM Products ORDER BY UnitPrice DESC LIMIT 1").with_inputs("question", "schema"),
    #dspy.Example(question="What is the total value of units in stock for all products?", schema=schema, sql_query="SELECT SUM(UnitPrice * UnitsInStock) FROM Products").with_inputs("question", "schema"),
    #dspy.Example(question="How many units are currently on order for product ID 5?", schema=schema, sql_query="SELECT UnitsOnOrder FROM Products WHERE ProductID = 5").with_inputs("question", "schema"),
    #dspy.Example(question="What is the maximum discount given in any order detail?", schema=schema, sql_query="SELECT MAX(Discount) FROM [Order Details]").with_inputs("question", "schema"),
    #dspy.Example(question="What is the lowest freight charge paid on an order?", schema=schema, sql_query="SELECT MIN(Freight) FROM Orders").with_inputs("question", "schema"),

    # --- Grouping (GROUP BY / HAVING) ---
    #dspy.Example(question="Count the number of customers in each country.", schema=schema, sql_query="SELECT Country, COUNT(*) FROM Customers GROUP BY Country").with_inputs("question", "schema"),
    #dspy.Example(question="Which suppliers have more than 5 products?", schema=schema, sql_query="SELECT SupplierID, COUNT(*) FROM Products GROUP BY SupplierID HAVING COUNT(*) > 5").with_inputs("question", "schema"),
    #dspy.Example(question="What is the average freight cost for each ship country?", schema=schema, sql_query="SELECT ShipCountry, AVG(Freight) FROM Orders GROUP BY ShipCountry").with_inputs("question", "schema"),
    #dspy.Example(question="How many orders did each employee handle?", schema=schema, sql_query="SELECT EmployeeID, COUNT(*) FROM Orders GROUP BY EmployeeID").with_inputs("question", "schema"),
    #dspy.Example(question="Find the total quantity sold for each product ID.", schema=schema, sql_query="SELECT ProductID, SUM(Quantity) FROM [Order Details] GROUP BY ProductID").with_inputs("question", "schema"),

    # --- Ordering & Date Logic ---
    #dspy.Example(question="List the top 5 most expensive products.", schema=schema, sql_query="SELECT ProductName, UnitPrice FROM Products ORDER BY UnitPrice DESC LIMIT 5").with_inputs("question", "schema"),
    #dspy.Example(question="Show orders placed after January 1st, 1998.", schema=schema, sql_query="SELECT OrderID, OrderDate FROM Orders WHERE OrderDate > '1998-01-01'").with_inputs("question", "schema"),
    #dspy.Example(question="List orders sorted by shipping date, most recent first.", schema=schema, sql_query="SELECT OrderID, ShippedDate FROM Orders ORDER BY ShippedDate DESC").with_inputs("question", "schema"),
    #dspy.Example(question="Find orders where the shipping fee is zero.", schema=schema, sql_query="SELECT OrderID FROM Orders WHERE Freight = 0").with_inputs("question", "schema"),

    # --- Joins (Linking Tables) ---
    #dspy.Example(question="What are the product names included in Order ID 10248?", schema=schema, sql_query="SELECT T2.ProductName FROM [Order Details] AS T1 JOIN Products AS T2 ON T1.ProductID = T2.ProductID WHERE T1.OrderID = 10248").with_inputs("question", "schema"),
    #dspy.Example(question="Get the company name of the customer who placed Order ID 10250.", schema=schema, sql_query="SELECT T2.CompanyName FROM Orders AS T1 JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.OrderID = 10250").with_inputs("question", "schema"),
    #dspy.Example(question="List the Order IDs and Dates for the customer 'Alfreds Futterkiste'.", schema=schema, sql_query="SELECT T1.OrderID, T1.OrderDate FROM Orders AS T1 JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.CompanyName = 'Alfreds Futterkiste'").with_inputs("question", "schema"),
    #dspy.Example(question="Calculate the total revenue for Order 10248.", schema=schema, sql_query="SELECT SUM(UnitPrice * Quantity) FROM [Order Details] WHERE OrderID = 10248").with_inputs("question", "schema"),
    #dspy.Example(question="List all products from Category ID 1.", schema=schema, sql_query="SELECT ProductName FROM Products WHERE CategoryID = 1").with_inputs("question", "schema"),
    dspy.Example(question="Show the distinct regions where customers are located.", schema=schema, sql_query="SELECT DISTINCT Region FROM Customers").with_inputs("question", "schema"),
]

# --- 3. Define the Evaluation Metric ---
# We'll check if the generated SQL exactly matches the gold standard query.
def sql_exact_match(example, pred, trace=None):
    return example.sql_query.lower() == pred.sql_query.lower()

# --- 4. Compare "Before" and "After" ---

# BEFORE: Evaluate the un-optimized module
unoptimized_text2sql = TextToSQL()
evaluator = Evaluate(devset=trainset, num_threads=1, display_progress=True, display_table=5)
print("--- Evaluating Unoptimized TextToSQL ---")
evaluator(unoptimized_text2sql, metric=sql_exact_match)

# AFTER: Optimize the module with BootstrapFewShot
def optimization():
    optimizer = BootstrapFewShot(metric=sql_exact_match, max_bootstrapped_demos=2)
    optimized_text2sql = optimizer.compile(TextToSQL(), trainset=trainset)
    return optimized_text2sql

print("\n--- Evaluating Optimized TextToSQL ---")
optimized_text2sql = optimization()
evaluator(optimized_text2sql, metric=sql_exact_match)

#print("\n--- Example Prediction from Optimized Model ---")
#test_question = "Which products have a unit price greater than 50?"
#prediction = optimized_text2sql(question=test_question, schema=schema)
#print(f"Question: {test_question}")
#print(f"Predicted SQL Query: {prediction.sql_query}")


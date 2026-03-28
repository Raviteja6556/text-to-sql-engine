
!pip install duckdb pandas
print("Libraries installed successfully.")

import kagglehub

path = kagglehub.dataset_download("bytadit/ecommerce-order-dataset")

print("Path to dataset files:", path)

"""## Load CSVs into Pandas and DuckDB

### Subtask:
Inspect the directory structure of the downloaded Kaggle dataset to determine the correct path to the CSV files.

"""

print(f"Listing contents of: {path}")
!ls -R {path}

"""## Setup Environment and Load Data

### Subtask:
Load CSV files into pandas DataFrames, then transfer these DataFrames into an in-memory DuckDB database, and verify the successful creation of tables in DuckDB.

"""

import duckdb
import pandas as pd
import os

# 1. Define dataset_parent_folder, dataset_subfolder, and base_path_for_csvs
dataset_parent_folder = 'Ecommerce Order Dataset'
dataset_subfolder = 'train'
base_path_for_csvs = os.path.join(path, dataset_parent_folder, dataset_subfolder)

print(f"Base path for CSVs: {base_path_for_csvs}")

# 2. Create a dictionary named csv_files
csv_files = {
    'customers': 'df_Customers.csv',
    'orders': 'df_Orders.csv',
    'order_items': 'df_OrderItems.csv',
    'payments': 'df_Payments.csv',
    'products': 'df_Products.csv'
}

# 3. Establish an in-memory DuckDB connection
con = duckdb.connect(database=':memory:', read_only=False)

# 4. Initialize an empty dictionary to store pandas DataFrames
dataframes = {}

print("Loading CSVs into pandas DataFrames and then into DuckDB...")

# 5. Iterate through the csv_files dictionary
for table_name, file_name in csv_files.items():
    full_path = os.path.join(base_path_for_csvs, file_name)
    print(f"Processing {file_name} as table {table_name}...")

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(full_path)
    dataframes[table_name] = df

    # Use the DuckDB connection to create a table from the pandas DataFrame
    # The DataFrame 'df' is available in the local scope of the loop
    con.execute(f'CREATE TABLE {table_name} AS SELECT * FROM df');
    print(f"Table '{table_name}' created in DuckDB.")

# 6. Verify that the tables have been successfully created in DuckDB
print("\nVerifying tables in DuckDB:")
print(con.execute('PRAGMA show_tables;').fetchdf())

print("CSVs loaded into pandas and DuckDB successfully.")

"""## Extract and Format Schema for RAG

### Subtask:
Connect to the DuckDB database and programmatically extract table names, column names, and their corresponding data types using DuckDB's PRAGMA table_info. Format this information into concise text snippets suitable for RAG. Explicitly print all generated schema descriptions for verification.

"""

import duckdb

# 1. Retrieve the list of all table names from the DuckDB connection con
table_names = con.execute('PRAGMA show_tables;').fetchdf()['name'].tolist()

# 2. Initialize an empty list called schema_descriptions
schema_descriptions = []

print("Extracting and formatting schema descriptions...")

# 3. Iterate through each table_name
for table_name in table_names:
    # a. For each table, execute PRAGMA table_info(<table_name>);
    column_info = con.execute(f"PRAGMA table_info('{table_name}');").fetchdf()

    columns = []
    # b. Iterate through the results of PRAGMA table_info to extract each column's name and its data type.
    for index, row in column_info.iterrows():
        col_name = row['name']
        col_type = row['type']
        # c. Format the column name and type into a string like '`column_name` (DATA_TYPE)'
        columns.append(f"`{col_name}` ({col_type})")

    # d. Join all formatted column strings for the current table with a semicolon and space (`; `).
    columns_str = '; '.join(columns)

    # e. Create a complete schema description string for the table
    table_description = f"Table `{table_name}` has columns: {columns_str}"

    # f. Append this formatted schema description string to the schema_descriptions list.
    schema_descriptions.append(table_description)

# 4. After processing all tables, print each schema description
print("\nGenerated Schema Descriptions:")
for desc in schema_descriptions:
    print(desc)

"""## Generate Schema Embeddings

### Subtask:
Install `sentence-transformers`, load the 'all-MiniLM-L6-v2' pre-trained embedding model, generate vector embeddings for each of the formatted schema descriptions, and verify the successful generation by printing the type and shape of the resulting embeddings.

"""

import sys
!{sys.executable} -m pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading 'all-MiniLM-L6-v2' model and generating embeddings...")

# 3. Load the pre-trained 'all-MiniLM-L6-v2' model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Generate embeddings for the schema_descriptions list
schema_embeddings = embedding_model.encode(schema_descriptions)

# 5. Print the type of schema_embeddings
print(f"\nType of schema_embeddings: {type(schema_embeddings)}")

# 6. Print the shape of schema_embeddings
print(f"Shape of schema_embeddings: {schema_embeddings.shape}")

print("Schema embeddings generated successfully.")

import sys
!{sys.executable} -m pip install chromadb

import chromadb

# 1. Define the path for the persistent ChromaDB store
CHROMA_PERSIST_PATH = "./chroma_store"

print(f"Initializing ChromaDB client at: {CHROMA_PERSIST_PATH}")

# 2. Initialize a persistent ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

# 3. Define a collection name
collection_name = "schema_embeddings"

# 4. Get or create the collection
# If the collection already exists, it will be returned; otherwise, a new one will be created.
print(f"Getting or creating collection: {collection_name}")
collection = client.get_or_create_collection(name=collection_name)

# 5. Prepare IDs for the schema descriptions (e.g., 'schema_0', 'schema_1', etc.)
ids = [f"schema_{i}" for i in range(len(schema_descriptions))]

# 6. Add the schema descriptions, embeddings, and IDs to the ChromaDB collection
print("Adding schema descriptions and embeddings to ChromaDB...")
collection.add(
    documents=schema_descriptions,
    embeddings=schema_embeddings.tolist(), # ChromaDB expects a list of lists
    ids=ids
)

# 7. Verify the number of items in the collection
count = collection.count()
print(f"Successfully added {count} items to the '{collection_name}' collection.")

import sys
!{sys.executable} -m pip install fastapi uvicorn langchain-google-genai python-dotenv

import os
from dotenv import load_dotenv
from google.colab import userdata

print("Libraries installed successfully for FastAPI and LLM integration.")

# Load environment variables from .env file
load_dotenv()

# Directly set the Google API Key using userdata.get for secrets
# Ensure to strip any potential extraneous quotes from the retrieved key
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY').strip("'\"")

print("Environment setup for FastAPI and LLM completed.")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import duckdb
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import numpy as np # Added import for numpy to check array types

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Global Resources (Initialized once) ---
# DuckDB Connection
# 'con' is already initialized from previous cells

# ChromaDB Client and Collection
CHROMA_PERSIST_PATH = "./chroma_store"
client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
collection_name = "schema_embeddings"
collection = client.get_or_create_collection(name=collection_name)

# Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# LLM (Google Generative AI)
# Updated model to 'gemini-2.5-flash' based on available models list and quota issues
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# Base URL for API (for local testing with uvicorn)
BASE_URL = "http://localhost:8000"

# Pydantic model for request body for SQL execution
class SQLQuery(BaseModel):
    sql_query: str

# Pydantic model for request body for question
class Question(BaseModel):
    question: str

# Pydantic model for request body for chat
class ChatMessage(BaseModel):
    message: str

# --- API Endpoints ---
@app.get("/", summary="Root endpoint for the API")
async def root():
    return {"message": "SQL Generation API is running!"}

@app.get("/schema", summary="Retrieve the database schema descriptions")
async def get_schema():
    """Returns the list of formatted database schema descriptions."""
    global schema_descriptions # Ensure schema_descriptions is accessible
    return {"schema_descriptions": schema_descriptions}


@app.post("/generate-sql", summary="Generate SQL query from natural language question")
async def generate_sql(question_data: Question):
    """Generates an SQL query based on a natural language question and relevant schema information."""
    user_question = question_data.question

    # 1. Embed the user's question
    question_embedding = embedding_model.encode([user_question]).tolist()

    # 2. Query ChromaDB for relevant schema descriptions
    # Using n_results=2 to get the top 2 most relevant schema descriptions
    retrieved_schemas = collection.query(
        query_embeddings=question_embedding,
        n_results=2
    )

    # Extract document content from the query results
    context = "\n".join(retrieved_schemas['documents'][0]) if retrieved_schemas['documents'] else "No relevant schema found."
    print(f"RAG Context: {context}") # Debugging print

    # 3. Construct the prompt for the LLM
    prompt = f"""Given the following database schema:
{context}

**IMPORTANT**:
- Only use table names and column names exactly as provided in the schema. Do not invent new ones.
- When joining 'payments' and 'products' tables, remember that 'order_items' table is used to link them.

Generate a DuckDB SQL query to answer the following question: "{user_question}"

Only return the SQL query, without any additional text or explanations.
"""

    print(f"LLM Prompt:\n{prompt}") # Debugging print

    # 4. Generate SQL using the LLM
    try:
        response = llm.invoke(prompt)
        sql_query = response.content.strip()
        # Remove markdown code block delimiters if present
        if sql_query.startswith("```sql") and sql_query.endswith("```"):
            sql_query = sql_query[len("```sql"): -len("```")].strip()
        elif sql_query.startswith("```") and sql_query.endswith("```"):
            sql_query = sql_query[len("```"): -len("```")].strip()

        print(f"LLM Generated SQL: {sql_query}") # Debugging print
        return {"sql_query": sql_query}
    except Exception as e:
        print(f"Error during LLM invocation in generate_sql: {e}") # Added debug print
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {e}")


@app.post("/execute-sql", summary="Execute an SQL query against the DuckDB database")
async def execute_sql(query_data: SQLQuery):
    """Executes a given SQL query against the DuckDB database and returns the results."""
    try:
        print(f"Executing SQL: {query_data.sql_query}") # Debugging print
        # 'con' is the global DuckDB connection object from previous cells
        result_df = con.execute(query_data.sql_query).fetchdf()

        # Custom serialization for columns that might contain non-standard types (like DuckDB LIST/ARRAY)
        processed_records = []
        for _, row in result_df.iterrows():
            record = row.to_dict()
            for key, value in record.items():
                # If a value is a list or numpy array, ensure its elements are basic types (like str)
                if isinstance(value, (list, np.ndarray)):
                    record[key] = [str(item) for item in value]
            processed_records.append(record)

        return {"results": processed_records}
    except Exception as e:
        print(f"Error during SQL execution: {e}") # Added debug print to capture detailed DuckDB error
        raise HTTPException(status_code=400, detail=f"Error executing SQL query: {e}")

@app.post("/chat", summary="Handle a natural language chat message, generate SQL, and execute it")
async def chat(chat_message: ChatMessage):
    """Takes a natural language message, generates an SQL query, executes it, and returns the results."""
    try:
        # Generate SQL from the chat message
        sql_response = await generate_sql(Question(question=chat_message.message))
        sql_query = sql_response["sql_query"]

        print(f"SQL Query sent for execution (from chat): {sql_query}") # Debugging print

        # Execute the generated SQL
        execution_response = await execute_sql(SQLQuery(sql_query=sql_query))
        results = execution_response["results"]

        return {
            "question": chat_message.message,
            "generated_sql": sql_query,
            "results": results
        }
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        print(f"Error during chat processing: {e}") # Added debug print
        raise HTTPException(status_code=500, detail=f"Chat processing error: {e}")

print("FastAPI app, DuckDB, ChromaDB, Embedding Model, and LLM initialized.")
print(f"API Base URL: {BASE_URL}")
print("'/schema', '/generate-sql', '/execute-sql', and '/chat' endpoints added.")

"""**Reasoning**:
The FastAPI server needs to be restarted for the changes made in the previous step (stripping markdown code block delimiters from the generated SQL) to take effect. This will allow the subsequent simulation of the Streamlit query to correctly execute the SQL.


"""

import uvicorn
import nest_asyncio
from threading import Thread

print("Setting up FastAPI server for Colab...")

nest_asyncio.apply()

# Function to run the FastAPI app in a separate thread
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start the FastAPI server in a new thread
api_thread = Thread(target=run_fastapi)
api_thread.start()

print(f"FastAPI server started at {BASE_URL}. You can now interact with the API.")
print("To stop the server, interrupt the kernel (Runtime > Interrupt execution).")

"""**Reasoning**:
The FastAPI server has been updated with the SQL markdown stripping logic and restarted. Now, I will re-run the Streamlit query simulation to confirm that the `400 Bad Request` and `Parser Error` are resolved and the system can generate and execute SQL queries successfully.


"""

import requests
import pandas as pd

# Define the FastAPI backend URL
FASTAPI_URL = "http://localhost:8000"

# 1. Choose a sample natural language question
sample_question = "What are the top 5 most expensive products?"

print(f"Sending question to FastAPI: '{sample_question}'\n")

try:
    # 2. Send a POST request to the /chat endpoint
    response = requests.post(
        f"{FASTAPI_URL}/chat",
        json={"message": sample_question}
    )
    response.raise_for_status() # Raise an exception for HTTP errors

    # 3. Capture and display the JSON response
    chat_data = response.json()
    print("Full JSON Response from FastAPI:")
    print(chat_data)
    print("\n---")

    # 4. Extract and print the 'generated_sql'
    generated_sql = chat_data.get("generated_sql")
    if generated_sql:
        print("Generated SQL Query:")
        print(generated_sql)
    else:
        print("No SQL query was generated.")

    print("\n---")

    # 5. Extract and display 'results' as a pandas DataFrame
    results = chat_data.get("results")
    if results is not None:
        if isinstance(results, list) and len(results) > 0:
            results_df = pd.DataFrame(results)
            print("Query Results:")
            print(results_df)
        else:
            print("Query executed successfully, but returned no results.")
    else:
        print("No results returned from the query execution.")

except requests.exceptions.RequestException as e:
    print(f"Error communicating with the FastAPI backend: {e}")
    print("Please ensure the FastAPI server is running in another cell or process.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



import requests
import pandas as pd

# Define the FastAPI backend URL
FASTAPI_URL = "http://localhost:8000"

# Get user input for the question
user_question = input("Please enter your question about the e-commerce data: ")

print(f"\nSending question to FastAPI: '{user_question}'\n")

try:
    # Send a POST request to the /chat endpoint
    response = requests.post(
        f"{FASTAPI_URL}/chat",
        json={"message": user_question}
    )
    response.raise_for_status() # Raise an exception for HTTP errors

    # Capture and display the JSON response
    chat_data = response.json()
    print("Full JSON Response from FastAPI:")
    print(chat_data)
    print("\n---")

    # Extract and print the 'generated_sql'
    generated_sql = chat_data.get("generated_sql")
    if generated_sql:
        print("Generated SQL Query:")
        print(generated_sql)
    else:
        print("No SQL query was generated.")

    print("\n---")

    # Extract and display 'results' as a pandas DataFrame
    results = chat_data.get("results")
    if results is not None:
        if isinstance(results, list) and len(results) > 0:
            results_df = pd.DataFrame(results)
            print("Query Results:")
            display(results_df) # Use display for better DataFrame formatting
        else:
            print("Query executed successfully, but returned no results.")
    else:
        print("No results returned from the query execution.")

except requests.exceptions.RequestException as e:
    print(f"Error communicating with the FastAPI backend: {e}")
    print("Please ensure the FastAPI server is running in another cell or process.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

### Data Analysis Key Findings

*   The Text-to-SQL system successfully translates natural language questions into executable DuckDB SQL queries for e-commerce data, leveraging a Retrieval-Augmented Generation (RAG) approach with ChromaDB and a `sentence-transformers` model for schema context, and the `gemini-2.5-flash` LLM for SQL generation.
*   The development involved seven key phases: data acquisition (Kaggle dataset via `kagglehub`), data loading (`pandas` to DuckDB), schema extraction (DuckDB `PRAGMA`), schema embedding (`sentence-transformers` with 'all-MiniLM-L6-v2'), vector store creation (`chromadb`), FastAPI backend setup (with endpoints like `/generate-sql`, `/execute-sql`, and `/chat`), and LLM integration (`langchain-google-genai`).
*   Significant challenges were encountered and successfully resolved:
    *   **LLM quota limitations** with `gemini-pro-latest` were overcome by switching to the `gemini-2.5-flash` model.
    *   **LLM output formatting issues**, where SQL was wrapped in markdown blocks, were fixed by implementing a markdown stripping logic in the FastAPI's `/generate-sql` endpoint.
    *   **Incorrect table name generation** by the LLM was addressed by enhancing the prompt with explicit directives to use only provided schema names and specific instructions for joining tables (e.g., using `order_items` to link `payments` and `products`).
    *   **FastAPI JSON serialization errors** with DuckDB's list/array types were resolved by adding custom serialization logic in the `/execute-sql` endpoint to convert complex data types to string representations.
*   The final system provides dynamic SQL execution against an in-memory DuckDB database, robust error handling for LLM output and data serialization, and a simulated conversational interface for interactive data querying.

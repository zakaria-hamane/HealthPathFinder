import json
import random
from neo4j import GraphDatabase
from multiprocessing import Pool

# Connect to the Neo4j database
url = "bolt://neo4j.het.io:7687"
username = "neo4j"
password = None  # Replace with your password if required
driver = GraphDatabase.driver(url, auth=(username, password))


# Function for a single query execution in a process
def query_paths(args):
    start_type, end_type, offset, batch_size = args
    with driver.session() as session:
        query = f"""
        MATCH (start:{start_type}), (end:{end_type})
        MATCH path = shortestPath((start)<-[*..5]-(end))
        RETURN path
        SKIP {offset} LIMIT {batch_size}
        """
        result = session.run(query)
        serialized_paths = []
        for record in result:
            path = record["path"]
            nodes = [node["name"] for node in path.nodes]
            edges = [rel.type for rel in path.relationships]
            serialized_path = ' '.join(sum(zip(nodes, edges + ['']), ()))
            serialized_paths.append(serialized_path)
        return serialized_paths


# Function to generate negative paths
def generate_negative_paths(positive_paths, end_type):
    negative_paths = []
    for path in positive_paths:
        elements = path.split()
        # Replace the last node with a random end_type node
        elements[-1] = f"{end_type}_RandomNode_{random.randint(1, 1000)}"
        negative_path = ' '.join(elements)
        negative_paths.append(negative_path)
    return negative_paths


# Function to extract paths using multiprocessing
def extract_paths_parallel(start_type, end_type, batch_size=100, num_processes=4, max_paths=1200):
    paths = []
    offset = 0
    with Pool(num_processes) as pool:
        while True:
            args = [(start_type, end_type, offset + i * batch_size, batch_size) for i in range(num_processes)]
            results = pool.map(query_paths, args)
            batch = [item for sublist in results for item in sublist]
            if not batch or len(paths) >= max_paths:
                break
            paths.extend(batch)
            offset += batch_size * num_processes
            print(f"Extracted {len(paths)} paths so far...")

            # Check if the number of extracted paths has reached the maximum limit
            if len(paths) >= max_paths:
                print("Reached maximum path limit.")
                break
    return paths[:max_paths]


# Extract paths and generate negative paths
positive_symptom_disease = extract_paths_parallel("Symptom", "Disease")
positive_disease_compound = extract_paths_parallel("Disease", "Compound")

negative_symptom_disease = generate_negative_paths(positive_symptom_disease, "Disease")
negative_disease_compound = generate_negative_paths(positive_disease_compound, "Compound")


def split_data(data, test_ratio=0.2):
    """
    Splits each list in the data into training and test sets.

    :param data: Dictionary of lists of paths
    :param test_ratio: Fraction of data to be used as test set
    :return: Tuple of dictionaries (train_set, test_set)
    """
    train_set = {}
    test_set = {}
    for key, value in data.items():
        random.shuffle(value)
        split_index = int(len(value) * (1 - test_ratio))
        train_set[key] = value[:split_index]
        test_set[key] = value[split_index:]
    return train_set, test_set

# Combine and format data
data = {
    "symptom_disease_positive": positive_symptom_disease,
    "disease_compound_positive": positive_disease_compound,
    "symptom_disease_negative": negative_symptom_disease,
    "disease_compound_negative": negative_disease_compound
}

# Split the data into train and test sets
train_data, test_data = split_data(data, test_ratio=0.2)

# Save as JSON
with open('train_data.json', 'w') as f:
    json.dump(train_data, f)
with open('test_data.json', 'w') as f:
    json.dump(test_data, f)

# Close the database connection
driver.close()

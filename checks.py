import socket
from opensearchpy import OpenSearch

# Replace with your OpenSearch server details
host = "172.31.30.137"
port = 9200
username = "admin"  # Replace with your username
password = "H@RTn311_ROCKS"  # Replace with your password

# Check if IP and port are reachable
def is_port_open(host, port, timeout=5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def OS_client():
    # Create the OpenSearch client with authentication
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(username, password),  # Basic Authentication
        use_ssl=False,  # Set to True if using HTTPS
        verify_certs=False  # Set to True if you have valid SSL certs
        ,timeout=30,
        retry_on_timeout=True,
        max_retries=3
    )

    # Test the OpenSearch connection
    print("started")
    # client.indices.delete(index='security-auditlog-*')
    try:
        response = client.info()
        print("Connected to OpenSearch:", response)
    except Exception as e:
        print("Error connecting to OpenSearch:", e)
    print("ended")
    # indices = client.indices.get('*')
    # for index in indices:
    #     # if index[0] != '.':

    #         client.indices.put_settings(
    #             index=index,
    #             body={"index": {"number_of_replicas": 0}}
    #         )

    health = client.cluster.health()
    print(health)
    # plugins_info = client.nodes.info(metric="plugins")
    # for node_id, node in plugins_info["nodes"].items():
    #     print(f"Node: {node['name']}")
    #     for plugin in node["plugins"]:
    #         print(f"  - {plugin['name']}")



    # indices = client.cat.indices(format="json")

    shards = client.cat.shards(format="json")
    print(len((shards)))
    for shard in shards:
        print(shard)

def wait_until_opensearch_ready(client, timeout=10):
    import time
    for i in range(timeout):
        try:
            health = client.cluster.health()
            if health["status"] in ("green", "yellow"):
                return
        except Exception as e:
            print(f"[{i+1}/{timeout}] Waiting for OpenSearch: {e}")
        time.sleep(1)
    raise RuntimeError("OpenSearch never became healthy")


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    print("Testing IP and port reachability...")
    if not is_port_open(host, port):
        print(f"Cannot reach {host}:{port}. Check network or server settings.")
    else:
        print(f"{host}:{port} is reachable. Proceeding with OpenSearch connection...")


    OS_client()


    # client = OpenSearch(
    #     hosts=[{"host": host, "port": port}],
    #     http_auth=(username, password),  # Basic Authentication
    #     use_ssl=True,  # Set to True if using HTTPS
    #     verify_certs=False  # Set to True if you have valid SSL certs
    #     ,timeout=30,
    #     retry_on_timeout=True,
    #     max_retries=3
    # )

    # wait_until_opensearch_ready(client=client)

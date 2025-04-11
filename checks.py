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

print("Testing IP and port reachability...")
if not is_port_open(host, port):
    print(f"Cannot reach {host}:{port}. Check network or server settings.")
else:
    print(f"{host}:{port} is reachable. Proceeding with OpenSearch connection...")

    # Create the OpenSearch client with authentication
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(username, password),  # Basic Authentication
        use_ssl=True,  # Set to True if using HTTPS
        verify_certs=False  # Set to True if you have valid SSL certs
        ,timeout=30,
        retry_on_timeout=True,
        max_retries=3
    )

    # Test the OpenSearch connection
    print("started")
    try:
        response = client.info()
        print("Connected to OpenSearch:", response)
    except Exception as e:
        print("Error connecting to OpenSearch:", e)
    print("ended")
    health = client.cluster.health()
    print(health)


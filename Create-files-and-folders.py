import os

# Define the folder structure
structure = {
    "backend": {
        "app": {
            "__init__.py": "",
            "main.py": "",
            "models": {
                "__init__.py": "",
                "user.py": "",
                "receipt.py": "",
            },
            "schemas": {
                "__init__.py": "",
                "user.py": "",
                "receipt.py": "",
            },
            "routes": {
                "__init__.py": "",
                "auth.py": "",
                "receipt.py": "",
            },
            "services": {
                "__init__.py": "",
                "auth.py": "",
                "receipt.py": "",
            },
            "utils": {
                "__init__.py": "",
                "security.py": "",
                "websocket.py": "",
            },
            "database.py": "",
            "config.py": "",
        },
        "requirements.txt": "",
        "README.md": "",
    }
}

# Function to create the folder structure
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write(content)

# Create the structure
create_structure(".", structure)
print("Folder structure created successfully!")
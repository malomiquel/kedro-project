import os
from flask import Flask
from api import create_app

# Ensure the correct working directory
project_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_path)

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)

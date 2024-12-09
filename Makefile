# Variables
FLASK_APP=app.py
FLASK_ENV=development
FLASK_RUN_PORT=5000

# Default target
run:
	flask run --port=$(FLASK_RUN_PORT)

# Install dependencies
install:
	pip install -r requirements.txt

# Check Python code format
lint:
	flake8 $(FLASK_APP)

# Clean .pyc files and cache
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Display help
help:
	@echo "Available commands:"
	@echo "  make run       Start the Flask application"
	@echo "  make install   Install dependencies"
	@echo "  make lint      Check code style"
	@echo "  make clean     Remove .pyc files and cache"

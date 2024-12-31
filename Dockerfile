# Docker command FROM specifies the base image of the container
# Our base image is Linux with python-3.11.8 preinstalled
FROM python:3.11.8

# Set the environment variable
ENV APP_HOME=/app

# Set the working directory inside the container
WORKDIR $APP_HOME

# Copy other files to the working directory of the container
COPY frontend/ .

COPY requirements.txt .

# Install dependencies inside the container
RUN pip install -r requirements.txt

# Mark the port where the application runs inside the container
EXPOSE 8501

# Let's run our application inside the container
ENTRYPOINT [“streamlit”, “run”, “app.py”]
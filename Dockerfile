# Use an official Python runtime as a parent image
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside the container
EXPOSE 5000

# Run app.py when the container launches
# CMD ["python", "src/app.py"]
CMD ["src.app.lambda_handler"]
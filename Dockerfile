# Use an official NVIDIA CUDA-enabled PyTorch image as a base
# This ensures PyTorch can use the GPU on your remote server
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install your project's dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Set a default command to run when the container starts
CMD ["python", "train.py"]
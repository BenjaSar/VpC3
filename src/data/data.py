import kagglehub

# Download latest version
path = kagglehub.dataset_download("qmarva/cubicasa5k")

print("Path to dataset files:", path)
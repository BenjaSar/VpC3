# vit-floorplan

Vision Transformer for floorplan segmentation and classification

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/BenjaSar/vit_floorplan.git
cd vit_floorplan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make dev-install
```

### ⭐ First Step: Run EDA Analysis
```bash
make eda
```

Or run directly:
```bash
python scripts/eda_analysis.py
```

### Training
```bash
make train
```

## 📁 Project Structure
```
vit_floorplan/
├── src/              # Source code
├── notebooks/        # Jupyter notebooks
├── tests/            # Unit tests
├── configs/          # Configuration files
├── scripts/          # Training and utility scripts
└── docker/           # Docker configuration
```

## 🔧 Development
```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint
```

## 🐳 Docker
```bash
# Build Docker image
make docker-build

# Run container
make docker-run
```

## 📊 MLFlow Tracking
```bash
mlflow ui
```

## 📝 License

MIT

## 👤 Author

**FS, Alejandro Lloveras, Jorge Cuenca**
- Email: fs@example.com
- GitHub: [@BenjaSar](https://github.com/BenjaSar)

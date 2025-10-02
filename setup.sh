#!/bin/bash
# Baby Cry Detection - Development Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🍼 Baby Cry Detection - Development Setup"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python 3.8+ is installed
echo -e "${BLUE}📋 Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.8+ is required. Current version: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Check if Flutter is installed (for mobile development)
echo -e "${BLUE}📋 Checking Flutter installation...${NC}"
if command -v flutter &> /dev/null; then
    FLUTTER_VERSION=$(flutter --version | head -1 | cut -d' ' -f2)
    echo -e "${GREEN}✅ Flutter $FLUTTER_VERSION found${NC}"
else
    echo -e "${YELLOW}⚠️  Flutter not found. Mobile development will be unavailable${NC}"
    echo -e "${YELLOW}   Install from: https://flutter.dev/docs/get-started/install${NC}"
fi

# Create virtual environment
echo -e "${BLUE}🐍 Creating Python virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}📦 Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install Python dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install -e .

# Create necessary directories
echo -e "${BLUE}📁 Creating project directories...${NC}"
mkdir -p data/{raw,processed,train,val,test}
mkdir -p models/{checkpoints,mobile}
mkdir -p data/train/{hunger,pain,discomfort,tired,normal}
mkdir -p data/val/{hunger,pain,discomfort,tired,normal}  
mkdir -p data/test/{hunger,pain,discomfort,tired,normal}

# Download datasets (optional)
echo -e "${BLUE}📊 Dataset Setup${NC}"
echo "Available datasets:"
echo "1. ICSD (Recommended) - 10,000+ samples"
echo "2. Donate A Cry - 457 samples"
echo "3. Skip dataset download"

read -p "Choose dataset to download (1-3): " -n 1 -r dataset_choice
echo

case $dataset_choice in
    1)
        echo -e "${BLUE}📥 Downloading ICSD dataset...${NC}"
        if command -v git &> /dev/null; then
            git clone https://github.com/QingyuLiu0521/ICSD.git data/raw/ICSD
            echo -e "${GREEN}✅ ICSD dataset downloaded${NC}"
            echo -e "${YELLOW}⚠️  You'll need to organize the data manually into train/val/test splits${NC}"
        else
            echo -e "${RED}❌ Git is required to download datasets${NC}"
        fi
        ;;
    2)
        echo -e "${BLUE}📥 Downloading Donate A Cry dataset...${NC}"
        if command -v git &> /dev/null; then
            git clone https://github.com/gveres/donateacry-corpus.git data/raw/donateacry
            echo -e "${GREEN}✅ Donate A Cry dataset downloaded${NC}"
        else
            echo -e "${RED}❌ Git is required to download datasets${NC}"
        fi
        ;;
    3)
        echo -e "${YELLOW}⏭️  Skipping dataset download${NC}"
        ;;
    *)
        echo -e "${YELLOW}⏭️  Invalid choice, skipping dataset download${NC}"
        ;;
esac

# Flutter app setup
if command -v flutter &> /dev/null; then
    echo -e "${BLUE}📱 Setting up Flutter app...${NC}"
    cd mobile
    flutter pub get
    cd ..
    echo -e "${GREEN}✅ Flutter dependencies installed${NC}"
fi

# Create .env file for development
echo -e "${BLUE}⚙️  Creating development configuration...${NC}"
cat > .env << EOF
# Baby Cry Detection - Development Configuration
PYTHON_ENV=development
MODEL_PATH=models/mobile/
DATA_PATH=data/
LOG_LEVEL=INFO
EOF

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activate the baby cry detection development environment
source venv/bin/activate
export PYTHONPATH=$PWD/ml-core:$PYTHONPATH
echo "🍼 Baby Cry Detection environment activated"
echo "Run 'deactivate' to exit the environment"
EOF

chmod +x activate_env.sh

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Activate environment: ${YELLOW}source activate_env.sh${NC}"
echo "2. Organize dataset: Move audio files to data/train/{class}/ folders"
echo "3. Train model: ${YELLOW}python cli.py train --model mobile --epochs 50${NC}"
echo "4. Convert model: ${YELLOW}python cli.py convert --checkpoint models/checkpoints/best_mobilecnn_babycry.pth --model mobile${NC}"
echo "5. Run mobile app: ${YELLOW}cd mobile && flutter run${NC}"
echo
echo -e "${BLUE}🔧 Development Commands:${NC}"
echo "• Train model: ${YELLOW}python cli.py train --model mobile${NC}"
echo "• Make predictions: ${YELLOW}python cli.py predict --audio test.wav --model [path]${NC}"
echo "• Convert to mobile: ${YELLOW}python cli.py convert --checkpoint [path] --model mobile${NC}"
echo "• Mobile app: ${YELLOW}cd mobile && flutter run${NC}"
echo
echo -e "${BLUE}📚 Documentation:${NC}"
echo "• README.md - Project overview and usage"
echo "• ml-core/baby_cry_detection/ - Python ML package"
echo "• mobile/ - Flutter mobile application"
echo
echo -e "${YELLOW}⚠️  Remember to organize your dataset before training!${NC}"
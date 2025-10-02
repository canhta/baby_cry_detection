#!/bin/bash
# Baby Cry Detection - Development Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🍼 Baby Cry Detection - Development Setup"
echo "========================================="

# Parse command line arguments
CLEAN_MODE=false
DEMO_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        --demo-data)
            DEMO_DATA=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --clean      Clean existing environment before setup"
            echo "  --demo-data  Copy demo audio files for immediate testing"
            echo "  --help, -h   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

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

# Cleanup if requested
if [ "$CLEAN_MODE" = true ]; then
    echo -e "${BLUE}🧹 Cleaning existing environment...${NC}"
    rm -rf venv
    rm -rf data/train/*/
    rm -rf data/val/*/
    rm -rf data/test/*/
    rm -rf models/checkpoints/*
    rm -rf models/mobile/*
    rm -f activate_env.sh
    rm -f .env
    echo -e "${GREEN}✅ Environment cleaned${NC}"
fi

# Create virtual environment
echo -e "${BLUE}🐍 Creating Python virtual environment...${NC}"
if [ -d "venv" ] && [ "$CLEAN_MODE" = false ]; then
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
echo "2. Skip dataset download"

read -p "Choose dataset to download (1-2): " -n 1 -r dataset_choice
echo

case $dataset_choice in
    1)
        echo -e "${BLUE}📥 Downloading ICSD dataset...${NC}"
        if command -v git &> /dev/null; then
            if [ -d "data/raw/ICSD" ]; then
                echo -e "${YELLOW}⚠️  ICSD dataset directory already exists${NC}"
                read -p "Do you want to update it? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo -e "${BLUE}🔄 Updating ICSD dataset...${NC}"
                    cd data/raw/ICSD
                    git pull origin main || git pull origin master
                    cd ../../..
                    echo -e "${GREEN}✅ ICSD dataset updated${NC}"
                else
                    echo -e "${GREEN}✅ Using existing ICSD dataset${NC}"
                fi
            else
                git clone https://github.com/QingyuLiu0521/ICSD.git data/raw/ICSD
                echo -e "${GREEN}✅ ICSD dataset downloaded${NC}"
            fi
            echo -e "${YELLOW}⚠️  You'll need to organize the data manually into train/val/test splits${NC}"
        else
            echo -e "${RED}❌ Git is required to download datasets${NC}"
        fi
        ;;
    2)
        echo -e "${YELLOW}⏭️  Skipping dataset download${NC}"
        ;;
    *)
        echo -e "${YELLOW}⏭️  Invalid choice, skipping dataset download${NC}"
        ;;
esac

# Setup demo data for immediate testing
setup_demo_data() {
    echo -e "${BLUE}🎵 Setting up demo data for testing...${NC}"
    
    # Check if demo files exist
    if [ -d "data/raw/ICSD/demo" ]; then
        echo -e "${BLUE}📁 Found demo files, copying to training directories...${NC}"
        
        # Copy demo files to train directories
        if [ -f "data/raw/ICSD/demo/Real_Infantcry.wav" ]; then
            cp "data/raw/ICSD/demo/Real_Infantcry.wav" "data/train/hunger/"
            cp "data/raw/ICSD/demo/Real_Infantcry.wav" "data/train/discomfort/"
            echo -e "${GREEN}✅ Copied Real_Infantcry.wav to hunger and discomfort${NC}"
        fi
        
        if [ -f "data/raw/ICSD/demo/Synth_Infantcry.wav" ]; then
            cp "data/raw/ICSD/demo/Synth_Infantcry.wav" "data/train/pain/"
            cp "data/raw/ICSD/demo/Synth_Infantcry.wav" "data/train/tired/"
            echo -e "${GREEN}✅ Copied Synth_Infantcry.wav to pain and tired${NC}"
        fi
        
        if [ -f "data/raw/ICSD/demo/Real_Snoring.wav" ]; then
            cp "data/raw/ICSD/demo/Real_Snoring.wav" "data/train/normal/"
            echo -e "${GREEN}✅ Copied Real_Snoring.wav to normal${NC}"
        fi
        
        # Copy to validation directories as well
        if [ -f "data/raw/ICSD/demo/Real_Infantcry.wav" ]; then
            cp "data/raw/ICSD/demo/Real_Infantcry.wav" "data/val/hunger/"
            cp "data/raw/ICSD/demo/Real_Infantcry.wav" "data/val/discomfort/"
        fi
        
        if [ -f "data/raw/ICSD/demo/Synth_Infantcry.wav" ]; then
            cp "data/raw/ICSD/demo/Synth_Infantcry.wav" "data/val/pain/"
            cp "data/raw/ICSD/demo/Synth_Infantcry.wav" "data/val/tired/"
        fi
        
        if [ -f "data/raw/ICSD/demo/Real_Snoring.wav" ]; then
            cp "data/raw/ICSD/demo/Real_Snoring.wav" "data/val/normal/"
        fi
        
        echo -e "${GREEN}✅ Demo data setup complete${NC}"
        echo -e "${GREEN}📊 Training data ready - you can now run: python baby_cry.py train --model mobile${NC}"
    else
        echo -e "${YELLOW}⚠️  Demo files not found in data/raw/ICSD/demo/${NC}"
        echo -e "${YELLOW}   Please download ICSD dataset first or add your own audio files${NC}"
    fi
}

# Setup demo data if requested or if no training data exists
if [ "$DEMO_DATA" = true ]; then
    setup_demo_data
else
    # Check if training directories are empty
    TRAIN_FILES=$(find data/train -name "*.wav" -o -name "*.mp3" -o -name "*.flac" 2>/dev/null | wc -l)
    if [ "$TRAIN_FILES" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  No training data found${NC}"
        read -p "Do you want to setup demo data for testing? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            setup_demo_data
        fi
    fi
fi

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
export PYTHONPATH=$PWD/ml_core:$PYTHONPATH
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

# Check if demo data was set up
TRAIN_COUNT=$(find data/train -name "*.wav" 2>/dev/null | wc -l)
if [ "$TRAIN_COUNT" -gt 0 ]; then
    echo "2. ${GREEN}✅ Demo data ready!${NC} Train model: ${YELLOW}python baby_cry.py train --model mobile --epochs 10${NC}"
    echo "3. Convert model: ${YELLOW}python baby_cry.py convert --checkpoint models/checkpoints/best_mobilecnn_babycry.pth --model mobile${NC}"
    echo "4. Run mobile app: ${YELLOW}cd mobile && flutter run${NC}"
else
    echo "2. Organize dataset: Move audio files to data/train/{class}/ folders"
    echo "3. Train model: ${YELLOW}python baby_cry.py train --model mobile --epochs 50${NC}"
    echo "4. Convert model: ${YELLOW}python baby_cry.py convert --checkpoint models/checkpoints/best_mobilecnn_babycry.pth --model mobile${NC}"
    echo "5. Run mobile app: ${YELLOW}cd mobile && flutter run${NC}"
fi

echo
echo -e "${BLUE}🔧 Development Commands:${NC}"
echo "• Train model: ${YELLOW}python baby_cry.py train --model mobile${NC}"
echo "• Train with specific device: ${YELLOW}python baby_cry.py train --model mobile --device mps${NC}"
echo "• Make predictions: ${YELLOW}python baby_cry.py predict --audio test.wav --model [path]${NC}"
echo "• Convert to mobile: ${YELLOW}python baby_cry.py convert --checkpoint [path] --model mobile${NC}"
echo "• Mobile app: ${YELLOW}cd mobile && flutter run${NC}"
echo
echo -e "${BLUE}🧹 Cleanup Commands:${NC}"
echo "• Clean setup: ${YELLOW}./setup.sh --clean${NC}"
echo "• Setup with demo data: ${YELLOW}./setup.sh --demo-data${NC}"
echo
echo -e "${BLUE}🖥️  Device Support:${NC}"
echo "• Auto detection: Automatically selects best available device"
echo "• Mac M1/M2: Uses MPS (Metal Performance Shaders) for acceleration"
echo "• NVIDIA GPU: Uses CUDA when available"
echo "• Fallback: CPU for compatibility"
echo
echo -e "${BLUE}📚 Documentation:${NC}"
echo "• README.md - Project overview and usage"
echo "• ml-core/baby_cry_detection/ - Python ML package"
echo "• mobile/ - Flutter mobile application"
echo
echo -e "${YELLOW}⚠️  Remember to organize your dataset before training!${NC}"
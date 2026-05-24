# SO100 Teleoperation & PPO Training with CLIP Integration

A real-time teleoperation and reinforcement learning system for the SO100 robotic arm in Webots simulation, featuring CLIP-based image analysis for vision-guided task completion.

## Features

- **Real-Time Teleoperation**: Manual control of SO100 robotic arm with keyboard interface
- **CLIP-Based Vision**: Uses CLIP embeddings for goal-oriented visual analysis
- **Performance Metrics**: Tracks similarity scores, task progress, and real-time feedback
- **PPO Training**: Reinforcement learning integration for autonomous task learning
- **Trial Management**: Automatic trial detection and data logging
- **CSV Logging**: Complete performance metrics tracking
- **Webots Integration**: Full compatibility with Webots robot simulation platform

## Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **Webots** installed and accessible from command line
- **Git** for cloning the repository
- GPU recommended for CLIP inference (CUDA-enabled GPU)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd recent-revise-PPO-baseline
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using Python venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import transformers; print('✓ All dependencies installed successfully')"
```

## Project Structure

```
recent-revise-PPO-baseline/
├── controllers/
│   ├── so100_tele/           # Teleoperation controller
│   └── so100_trainer/        # PPO training controller
├── envs/                     # Environment definitions
├── camera/                   # Camera utilities
├── utils/                    # Helper utilities
├── visualization/            # Visualization tools
├── worlds/                   # Webots world files
├── trial_logging/            # Trial data and logs
├── plots/                    # Generated plots
├── checkpoints/              # Model checkpoints
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Running Locally

### Option 1: Teleoperation Mode (Manual Control)

1. Open Webots and load a world file from `worlds/` directory
2. Run the teleoperation controller:

```bash
python controllers/so100_tele/so100_tele.py
```

**Keyboard Controls:**
- **Q/A**: Joint 1 (base rotation)
- **W/S**: Joint 2 (shoulder pitch)
- **E/D**: Joint 3 (elbow pitch)
- **R/F**: Joint 4 (wrist pitch)
- **T/G**: Joint 5 (wrist yaw)
- **Y/H**: Joint 6 (wrist roll)

### Option 2: Training Mode (PPO Learning)

```bash
python controllers/so100_trainer/train.py
```

This will:
- Initialize the PPO training environment
- Begin training episodes
- Log metrics to `trial_logging/` directory
- Save checkpoints to `checkpoints/` directory

### Option 3: Direct Python Testing

```bash
# Test imports and setup
python -c "from envs import *; from utils import *; print('Ready to run')"
```

## Configuration

### Environment Variables

Set custom paths or configurations by creating a `.env` file:

```bash
WEBOTS_HOME=/path/to/webots
DATA_DIR=./trial_logging
CHECKPOINT_DIR=./checkpoints
```

### Hyperparameters

Edit controller-specific settings in:
- `controllers/so100_tele/so100_tele.py` - Teleoperation parameters
- `controllers/so100_trainer/train.py` - PPO training hyperparameters

## Output Files

### Trial Data
- **CSV Logs**: `trial_logging/*.csv` - Trial metrics (similarity, distance, progress)
- **Plots**: `plots/` - Generated performance visualizations

### Model Checkpoints
- **Saved Models**: `checkpoints/` - PPO model checkpoints for resuming training

## Troubleshooting

### Issue: Webots Connection Error
**Solution**: Ensure Webots is running and the world file is loaded before starting the controller.

### Issue: CLIP Model Download Fails
**Solution**: The first run downloads ~1GB of model weights. Ensure stable internet connection. Models are cached in `~/.cache/huggingface/`

### Issue: GPU Memory Error
**Solution**: Reduce batch size or use CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

### Issue: Import Errors
**Solution**: Verify virtual environment is activated and dependencies installed:
```bash
pip install -r requirements.txt --upgrade
```

## Dependencies

Core packages included:
- **PyTorch**: Deep learning framework
- **Transformers**: CLIP and language models
- **NumPy/SciPy**: Scientific computing
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Pillow**: Image processing

See `requirements.txt` for complete version specifications.

## Development

### Adding New Features
1. Create feature branch: `git checkout -b feature/your-feature`
2. Test thoroughly with teleoperation and training modes
3. Update this README with new features/instructions
4. Commit and push changes

### Debugging
Enable verbose logging by modifying controller files:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

- **Faster Inference**: Use GPU acceleration (CUDA-compatible GPU required)
- **Memory Optimization**: Adjust batch sizes in training config
- **Data Logging**: Disable CSV logging for faster iteration:
  ```python
  ENABLE_LOGGING = False  # In controller
  ```

## License

[Add your license here - e.g., MIT, Apache 2.0, etc.]

## Support

For issues or questions:
1. Check Webots official documentation: https://cyberbotics.com/
2. Review PyTorch/Transformers documentation
3. Check project documentation files or logs in `trial_logging/`

## Citation

If you use this project in research, please cite:
```bibtex
[Add citation details if applicable]
```

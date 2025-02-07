# Handwriting Digit Recognition

A web-based application that recognizes handwritten digits using a deep learning model.

## Requirements

### Python Version
- Python 3.7 or higher

### Required Packages
Install all required packages using:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install tensorflow
pip install numpy
pip install opencv-python
pip install flask
pip install scipy
```

## Setup and Usage

1. **Clone the Repository**
```bash
git clone [repository-url]
cd [repository-name]
```

2. **Train the Model**
```bash
python train_model.py
```
This will:
- Load and preprocess the MNIST dataset
- Train the neural network
- Save the trained model as `handwriting_model.h5`
- Training typically takes 5-10 minutes depending on your hardware

3. **Start the Server**
```bash
python server.py
```
The server will start at `http://localhost:5000`

4. **Use the Application**
- Open your web browser and go to `http://localhost:5000`
- Draw a digit (0-9) on the canvas using your mouse
- Click "Recognize" to get the prediction
- Use "Clear" to erase and try another digit

## File Structure
```
.
├── README.md
├── requirements.txt
├── train_model.py    # Script to train the neural network
├── server.py         # Flask server
├── ai.py            # AI model interface
└── index.html       # Web interface
```

## Troubleshooting

### Common Issues

1. **Model Not Found Error**
   ```
   Error: Model file 'handwriting_model.h5' not found!
   ```
   Solution: Run `python train_model.py` first

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'tensorflow'
   ```
   Solution: Run `pip install -r requirements.txt`

3. **Port Already in Use**
   ```
   Error: Port 5000 is already in use
   ```
   Solution: Stop other applications using port 5000 or modify the port in `server.py`

### Performance Tips
- For better recognition:
  - Draw digits clearly and centered
  - Use thick lines
  - Draw digits similar to how they appear in textbooks
- Clear the canvas between attempts
- Ensure the digit fills a good portion of the canvas
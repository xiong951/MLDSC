# MLDSC-Machine learning integrated differential scanning calorimetry analysis for a semicrystalline homopolymer
# A web platform for automated DSC analysis.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/flask-app-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A web-based service for processing Differential Scanning Calorimetry (DSC) data. This project provides an automated pipeline for baseline correction, mathematical modeling，peak decomposition, and parameters calculation.

---

## ✨ Key Features
* **Automated Baseline Correction**: Iterative polynomial fitting and intelligent region extraction.
* **Peak Decomposition**: Separate overlapping endothermic/exothermic peaks accurately.
* **Mathematical Modeling**: Built-in support for Gaussian and EMG mixture models.
* **High Performance Backend**: Multiprocessing enabled for heavy data processing tasks.
* **Ready-to-use API**: Flask-based RESTful endpoints for easy integration with front-end applications.

---

## 📂 Repository Structure

Below is a breakdown of the core files and their specific functions within the system:

* **`app.py`**: The main entry point. Initializes the Flask server, handles routing, and manages multiprocessing for background data calculations. 
* **Algorithm Modules**: `BaselineCorrection.py`, `PeakDecomposition.py`, `ExpGaussMix.py`, etc. These files handle the core mathematical modeling, gradient descent optimization, and rendering of physical metrics.
* **`templates/`**: Contains all the frontend HTML files (e.g., `base.html`, `MLDSC.html`, `multi_file_process.html`). These Bootstrap-based templates construct the user interface for file uploading, batch processing, and displaying results.
* **`requirements.txt`**: Lists all necessary Python dependencies to run the server.

---

## 🛠️ Installation & Setup

Follow these steps to get the server running on your local machine.

### 1. Prerequisites
Ensure all required dependencies listed in `requirements.txt` have been successfully installed.
### 2. Download all the code of the Repository

### 3. Starting the Application
Start the Flask backend server locally (binds to http://127.0.0.1:5000/ by default).

## 📬 Feedback & Support

Currently, the data parsing capabilities of our script are tailored to standard DSC formats. However, we are always eager to expand our compatibility to accommodate the diverse needs of the scientific community. 

Should you have suggestions for integrating new raw data formats, or any general feedback regarding your experience with the platform, we would be delighted to hear from you. Your insights are invaluable in helping us refine and enhance this tool. All feedback from the users is very welcome.



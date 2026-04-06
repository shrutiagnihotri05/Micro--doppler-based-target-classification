# Micro-Doppler Target Classification (Drone vs. Bird)

## Overview
This project aims to classify moving targets—specifically distinguishing between drones and birds—using their micro-Doppler radar signatures. 

Currently, this repository contains the Phase 1 and Phase 2 implementations, which focus on data pre-processing, Exploratory Data Analysis (EDA), and time-frequency feature extraction using the Short-Time Fourier Transform (STFT).

## Current Features

* **Data Cleaning & EDA (`data_cleaning.py`):** * Loads the synthetic micro-Doppler dataset.
  * Handles missing values using forward and backward filling.
  * Standardizes the radar signal amplitudes using `StandardScaler`.
  * Generates visualizations of the class distribution and compares the 1D time-domain signals of both classes.

* **Feature Extraction (`short_time_fourier_transform.py`):**
  * Converts the 1D time-series radar signals into 2D micro-Doppler spectrograms.
  * Utilizes `scipy.signal.stft` to extract frequency signatures over time.
  * Plots comparative spectrograms for both drones (Label 1) and birds (Label 0) to visually highlight the differences in their micro-Doppler kinematics.

## Project Structure
```text
├── Datasets/
│   └── synthetic_micro_doppler_dataset.csv   # Raw time-series radar data
├── data_cleaning.py                          # Script for cleaning and EDA
├── short_time_fourier_transform.py           # Script for generating STFT spectrograms
└── README.md
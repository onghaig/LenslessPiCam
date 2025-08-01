import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
from sympy import beta


def stokes_from_alphabeta(alpha,beta):
    # fast axis of HWP and QWP should be vertical
    # alpha: an array of HWP angels, in rad
    # beta: an array of QEP angels, in rad
    # return array(n,3), each row represent [S1/S0,S2/S0,S3/S0]
    result = np.zeros((alpha.shape[0],3),dtype=np.float32)
    result[:,0] = 0.5*(np.cos(4*alpha-4*beta)+np.cos(4*alpha))
    result[:,1] = 0.5*(np.sin(4*alpha)-np.sin(4*alpha-4*beta))
    result[:,2] = -np.sin(4*alpha-2*beta)
    return result

parser = argparse.ArgumentParser(description="Plot Angle vs Green Channel Output from CSV.")
parser.add_argument("csv_path", help="Path to the CSV file")
args = parser.parse_args()
csv_path = args.csv_path

# Read the first line to get exposure value and column names
with open(csv_path, "r") as f:
    header = f.readline()
    match = re.search(r"exp\s*=\s*([0-9.]+)", header)
    exp_val = match.group(1) if match else "unknown"
    # Remove the exp part for column names
    columns = [col.strip() for col in header.split(",") if "exp" not in col]

# Read the data, skipping the first line (header with exp)
df = pd.read_csv(csv_path)
beta = df["angle_deg"]/180*np.pi
alpha = np.zeros_like(beta)

# Calculate Stokes parameters

stokes = stokes_from_alphabeta(alpha,beta)
S3 = stokes[:,2]


idxs = np.argsort(S3)
S3 = S3[idxs]
intensities = df["ch1"].values[idxs]


# Plot angle vs ch1 (Green channel)
plt.plot(S3, intensities, marker='o')
plt.xlabel("S3")
plt.ylabel("Green Channel Output (ch1)")
plt.title(f"Angle vs Green with exp == {exp_val}")
plt.grid(True)
plt.show()
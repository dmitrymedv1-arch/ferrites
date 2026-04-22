"""
BaFeO₃ Multi-Property Predictor for Doped Ferrites - Version 1.2
Advanced tool for analyzing and predicting electrical conductivity, ASR,
power output, and thermal expansion of BaFeO₃-based perovskites.
Uses electronegativity-based descriptors instead of ionic radii.
OPTIMIZED VERSION: Fixed NaN handling, physical constraints, robust interpolation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.inspection import PartialDependenceDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
import shap
import warnings
from itertools import combinations
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.interpolate import griddata
import joblib
from datetime import datetime
import time
import plotly.figure_factory as ff
import io

warnings.filterwarnings('ignore')

# =============================================================================
# Modern scientific color palette and styling (SAME AS ORIGINAL)
# =============================================================================
MODERN_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'yellow': '#bcbd22',
    'cyan': '#17becf',
    'background': '#f8f9fa',
    'text': '#212529',
    'grid': '#dee2e6'
}

# Set modern plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})

# =============================================================================
# Electronegativity database (Pauling scale only)
# =============================================================================
ELECTRONEGATIVITY = {
    'H': 2.20, 'He': None, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': None, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': None,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.00,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16,
    'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10, 'I': 2.66, 'Xe': 2.60,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Pm': None, 'Sm': 1.17, 'Eu': None, 'Gd': 1.20, 'Tb': None, 'Dy': 1.22,
    'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': None, 'Lu': 1.27, 'Hf': 1.30,
    'Ta': 1.50, 'W': 2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28,
    'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'K': 0.82,
    'Na': 0.93, 'Li': 0.98, 'Rb': 0.82, 'Cs': 0.79, 'Mg': 1.31, 'F': 3.98,
    'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'S': 2.58, 'N': 3.04, 'P': 2.19,
}

# =============================================================================
# Additional properties for B-site cations (polarizability, mass)
# =============================================================================
B_SITE_PROPERTIES = {
    'Fe': {'polarizability': 0.18, 'mass': 55.845, 'std_oxidation': 3},
    'Zn': {'polarizability': 0.28, 'mass': 65.38, 'std_oxidation': 2},
    'Zr': {'polarizability': 0.37, 'mass': 91.224, 'std_oxidation': 4},
    'Y': {'polarizability': 0.55, 'mass': 88.906, 'std_oxidation': 3},
    'Ce': {'polarizability': 0.43, 'mass': 140.116, 'std_oxidation': 4},
    'Cu': {'polarizability': 0.15, 'mass': 63.546, 'std_oxidation': 2},
    'Ni': {'polarizability': 0.13, 'mass': 58.693, 'std_oxidation': 2},
    'Co': {'polarizability': 0.15, 'mass': 58.933, 'std_oxidation': 2},
    'Sn': {'polarizability': 0.24, 'mass': 118.710, 'std_oxidation': 4},
    'Bi': {'polarizability': 0.35, 'mass': 208.980, 'std_oxidation': 3},
    'Pr': {'polarizability': 0.42, 'mass': 140.908, 'std_oxidation': 3},
    'Nd': {'polarizability': 0.41, 'mass': 144.242, 'std_oxidation': 3},
    'Sm': {'polarizability': 0.40, 'mass': 150.36, 'std_oxidation': 3},
    'Gd': {'polarizability': 0.39, 'mass': 157.25, 'std_oxidation': 3},
    'La': {'polarizability': 1.04, 'mass': 138.905, 'std_oxidation': 3},
    'Sr': {'polarizability': 0.86, 'mass': 87.62, 'std_oxidation': 2},
    'Ba': {'polarizability': 1.55, 'mass': 137.327, 'std_oxidation': 2},
    'Ca': {'polarizability': 0.47, 'mass': 40.078, 'std_oxidation': 2},
    'Mg': {'polarizability': 0.09, 'mass': 24.305, 'std_oxidation': 2},
    'Al': {'polarizability': 0.05, 'mass': 26.982, 'std_oxidation': 3},
    'Ga': {'polarizability': 0.16, 'mass': 69.723, 'std_oxidation': 3},
    'In': {'polarizability': 0.42, 'mass': 114.818, 'std_oxidation': 3},
    'Sc': {'polarizability': 0.29, 'mass': 44.956, 'std_oxidation': 3},
    'Yb': {'polarizability': 0.31, 'mass': 173.045, 'std_oxidation': 3},
    'Er': {'polarizability': 0.32, 'mass': 167.259, 'std_oxidation': 3},
    'Dy': {'polarizability': 0.33, 'mass': 162.500, 'std_oxidation': 3},
    'Ho': {'polarizability': 0.32, 'mass': 164.930, 'std_oxidation': 3},
    'Tm': {'polarizability': 0.31, 'mass': 168.934, 'std_oxidation': 3},
    'Lu': {'polarizability': 0.30, 'mass': 174.967, 'std_oxidation': 3},
    'Hf': {'polarizability': 0.34, 'mass': 178.49, 'std_oxidation': 4},
    'Nb': {'polarizability': 0.22, 'mass': 92.906, 'std_oxidation': 5},
    'Ta': {'polarizability': 0.22, 'mass': 180.948, 'std_oxidation': 5},
    'Mo': {'polarizability': 0.20, 'mass': 95.95, 'std_oxidation': 6},
    'W': {'polarizability': 0.21, 'mass': 183.84, 'std_oxidation': 6},
    'Ti': {'polarizability': 0.19, 'mass': 47.867, 'std_oxidation': 4},
    'V': {'polarizability': 0.18, 'mass': 50.942, 'std_oxidation': 5},
    'Cr': {'polarizability': 0.17, 'mass': 51.996, 'std_oxidation': 3},
    'Mn': {'polarizability': 0.16, 'mass': 54.938, 'std_oxidation': 2},
}

# =============================================================================
# A-site properties (for A-site substitution)
# =============================================================================
A_SITE_PROPERTIES = {
    'Ba': {'polarizability': 1.55, 'mass': 137.327, 'std_oxidation': 2, 'chi': 0.89},
    'Sr': {'polarizability': 0.86, 'mass': 87.62, 'std_oxidation': 2, 'chi': 0.95},
    'Ca': {'polarizability': 0.47, 'mass': 40.078, 'std_oxidation': 2, 'chi': 1.00},
    'La': {'polarizability': 1.04, 'mass': 138.905, 'std_oxidation': 3, 'chi': 1.10},
    'Pr': {'polarizability': 0.42, 'mass': 140.908, 'std_oxidation': 3, 'chi': 1.13},
    'Nd': {'polarizability': 0.41, 'mass': 144.242, 'std_oxidation': 3, 'chi': 1.14},
    'Sm': {'polarizability': 0.40, 'mass': 150.36, 'std_oxidation': 3, 'chi': 1.17},
    'Gd': {'polarizability': 0.39, 'mass': 157.25, 'std_oxidation': 3, 'chi': 1.20},
    'Y': {'polarizability': 0.55, 'mass': 88.906, 'std_oxidation': 3, 'chi': 1.22},
    'Bi': {'polarizability': 0.35, 'mass': 208.980, 'std_oxidation': 3, 'chi': 2.02},
    'K': {'polarizability': 0.83, 'mass': 39.098, 'std_oxidation': 1, 'chi': 0.82},
    'Na': {'polarizability': 0.41, 'mass': 22.990, 'std_oxidation': 1, 'chi': 0.93},
    'Li': {'polarizability': 0.07, 'mass': 6.941, 'std_oxidation': 1, 'chi': 0.98},
    'Mg': {'polarizability': 0.09, 'mass': 24.305, 'std_oxidation': 2, 'chi': 1.31},
}

# =============================================================================
# Required columns for uploaded Excel file
# =============================================================================
REQUIRED_COLUMNS = [
    'no. paper', 'Composition', 'A\'', 'A\'\'', 'B\'', 'B\'\'', 'B\'\'\'', 'B\'\'\'"',
    'x', 'y', 'z', 'α', 'a (ox)', 'b (ox)', 'c (ox)', 'V (ox)', 'Vpseud (ox)',
    'apseud (ox)', 'σ (500 °C)', 'σ (600 °C)', 'σ (700 °C)', 'σmax',
    'αLT', 'αHT', 'αav', 'P(FC), 600 °C', 'P(FC), 650 °C', 'P(FC), 700 °C',
    'ASR, 600 °C', 'ASR, 650 °C', 'ASR, 700 °C', 'doi'
]

# =============================================================================
# Load uploaded Excel file (FIXED: better NaN handling)
# =============================================================================
@st.cache_data
def load_uploaded_excel(uploaded_file):
    """Load and validate uploaded Excel file with required columns"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        
        # Check required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols[:10]}...")
            return None
        
        # Clean data: replace '-' and empty strings with NaN
        df = df.replace(['-', '—', '', ' '], np.nan)
        
        # Convert numeric columns
        numeric_cols = ['x', 'y', 'z', 'α', 'a (ox)', 'b (ox)', 'c (ox)', 'V (ox)',
                        'Vpseud (ox)', 'apseud (ox)', 'σ (500 °C)', 'σ (600 °C)',
                        'σ (700 °C)', 'σmax', 'αLT', 'αHT', 'αav',
                        'P(FC), 600 °C', 'P(FC), 650 °C', 'P(FC), 700 °C',
                        'ASR, 600 °C', 'ASR, 650 °C', 'ASR, 700 °C']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # =========================================================
        # FIX 1: Очистка ASR от отрицательных значений
        # =========================================================
        asr_cols = ['ASR, 600 °C', 'ASR, 650 °C', 'ASR, 700 °C']
        for col in asr_cols:
            if col in df.columns:
                # Отрицательные значения заменяем на NaN
                df.loc[df[col] < 0, col] = np.nan
                # Нефизично большие значения (>1000 Ом·см²) тоже заменяем на NaN
                df.loc[df[col] > 1000, col] = np.nan
        
        # =========================================================
        # FIX 2: Очистка αLT и αHT от нулевых и отрицательных значений
        # =========================================================
        if 'αLT' in df.columns:
            df.loc[df['αLT'] <= 0, 'αLT'] = np.nan
        if 'αHT' in df.columns:
            df.loc[df['αHT'] <= 0, 'αHT'] = np.nan
        
        # =========================================================
        # FIX 3: Очистка проводимости от отрицательных значений
        # =========================================================
        sigma_cols = ['σ (500 °C)', 'σ (600 °C)', 'σ (700 °C)', 'σmax']
        for col in sigma_cols:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# =============================================================================
# Enhanced descriptor calculation functions (FIXED: NaN handling)
# =============================================================================
def get_electronegativity(element):
    """Get Pauling electronegativity for element"""
    if element is None or element == '-' or element == '' or pd.isna(element):
        return None
    return ELECTRONEGATIVITY.get(element, None)

def get_a_site_chi(A_prime, A_double_prime, x):
    """Calculate average A-site electronegativity: χ(A) = χ(A')*(1-x) + χ(A'')*x"""
    chi_Ap = get_electronegativity(A_prime) if A_prime != '-' and not pd.isna(A_prime) else 0
    chi_App = get_electronegativity(A_double_prime) if A_double_prime != '-' and not pd.isna(A_double_prime) else 0
    
    if chi_Ap is None:
        chi_Ap = 0
    if chi_App is None:
        chi_App = 0
    
    return chi_Ap * (1 - x) + chi_App * x

def get_b_site_chi(B_prime, B_double_prime, B_triple_prime, B_quad_prime, y, alpha):
    """
    Calculate average B-site electronegativity:
    χ(B) = (1-y-α)*χ(B') + y*χ(B'') + α*χ(B''')
    """
    chi_Bp = get_electronegativity(B_prime) if B_prime != '-' and not pd.isna(B_prime) else 0
    chi_Bpp = get_electronegativity(B_double_prime) if B_double_prime != '-' and not pd.isna(B_double_prime) else 0
    chi_Bppp = get_electronegativity(B_triple_prime) if B_triple_prime != '-' and not pd.isna(B_triple_prime) else 0
    
    if chi_Bp is None:
        chi_Bp = 0
    if chi_Bpp is None:
        chi_Bpp = 0
    if chi_Bppp is None:
        chi_Bppp = 0
    
    weight_Bp = 1 - y - alpha
    weight_Bpp = y
    weight_Bppp = alpha
    
    # Ensure weights are non-negative
    weight_Bp = max(0, weight_Bp)
    
    total_weight = weight_Bp + weight_Bpp + weight_Bppp
    if total_weight > 0:
        return (weight_Bp * chi_Bp + weight_Bpp * chi_Bpp + weight_Bppp * chi_Bppp) / total_weight
    return 0

def calculate_chi_diff(chi_A, chi_B):
    """Calculate |χ(A) - χ(B)|"""
    return abs(chi_A - chi_B)

def calculate_chi_product(chi_A, chi_B):
    """Calculate χ(A) * χ(B)"""
    return chi_A * chi_B

def calculate_oxygen_vacancy(y, alpha):
    """Calculate oxygen vacancy concentration from acceptor doping"""
    return (y + alpha) / 2

def calculate_average_polarizability(A_prime, A_double_prime, B_prime, B_double_prime, B_triple_prime, x, y, alpha):
    """Calculate weighted average polarizability of A and B sites"""
    # A-site polarizability
    pol_Ap = A_SITE_PROPERTIES.get(A_prime, {}).get('polarizability', 0) if A_prime != '-' and not pd.isna(A_prime) else 0
    pol_App = A_SITE_PROPERTIES.get(A_double_prime, {}).get('polarizability', 0) if A_double_prime != '-' and not pd.isna(A_double_prime) else 0
    pol_A_avg = pol_Ap * (1 - x) + pol_App * x
    
    # B-site polarizability
    pol_Bp = B_SITE_PROPERTIES.get(B_prime, {}).get('polarizability', 0) if B_prime != '-' and not pd.isna(B_prime) else 0
    pol_Bpp = B_SITE_PROPERTIES.get(B_double_prime, {}).get('polarizability', 0) if B_double_prime != '-' and not pd.isna(B_double_prime) else 0
    pol_Bppp = B_SITE_PROPERTIES.get(B_triple_prime, {}).get('polarizability', 0) if B_triple_prime != '-' and not pd.isna(B_triple_prime) else 0
    
    weight_Bp = max(0, 1 - y - alpha)
    weight_Bpp = y
    weight_Bppp = alpha
    total_weight = weight_Bp + weight_Bpp + weight_Bppp
    
    if total_weight > 0:
        pol_B_avg = (weight_Bp * pol_Bp + weight_Bpp * pol_Bpp + weight_Bppp * pol_Bppp) / total_weight
    else:
        pol_B_avg = 0
    
    return pol_A_avg, pol_B_avg, (pol_A_avg + pol_B_avg) / 2

def calculate_descriptors(row):
    """Calculate enhanced electronegativity-based descriptors for ferrites"""
    
    # Extract composition parameters with safe handling of NaN
    A_prime = row.get("A'", '-')
    if pd.isna(A_prime):
        A_prime = '-'
    
    A_double_prime = row.get("A''", '-')
    if pd.isna(A_double_prime):
        A_double_prime = '-'
    
    B_prime = row.get("B'", '-')
    if pd.isna(B_prime):
        B_prime = '-'
    
    B_double_prime = row.get("B''", '-')
    if pd.isna(B_double_prime):
        B_double_prime = '-'
    
    B_triple_prime = row.get("B'''", '-')
    if pd.isna(B_triple_prime):
        B_triple_prime = '-'
    
    B_quad_prime = row.get("B''''", '-')
    if pd.isna(B_quad_prime):
        B_quad_prime = '-'
    
    x = row.get('x', 0) if pd.notna(row.get('x', 0)) else 0
    y = row.get('y', 0) if pd.notna(row.get('y', 0)) else 0
    z = row.get('z', 1) if pd.notna(row.get('z', 1)) else 1
    alpha = row.get('α', 0) if pd.notna(row.get('α', 0)) else 0
    
    # Volume descriptors
    V_ox = row.get('V (ox)', np.nan) if pd.notna(row.get('V (ox)', np.nan)) else np.nan
    Vpseud = row.get('Vpseud (ox)', np.nan) if pd.notna(row.get('Vpseud (ox)', np.nan)) else np.nan
    apseud = row.get('apseud (ox)', np.nan) if pd.notna(row.get('apseud (ox)', np.nan)) else np.nan
    
    descriptors = {}
    
    # Electronegativity descriptors
    chi_A = get_a_site_chi(A_prime, A_double_prime, x)
    chi_B = get_b_site_chi(B_prime, B_double_prime, B_triple_prime, B_quad_prime, y, alpha)
    
    descriptors['chi_A'] = chi_A if chi_A is not None else 0
    descriptors['chi_B'] = chi_B if chi_B is not None else 0
    descriptors['chi_diff'] = calculate_chi_diff(chi_A, chi_B) if chi_A and chi_B else 0
    descriptors['chi_product'] = calculate_chi_product(chi_A, chi_B) if chi_A and chi_B else 0
    
    # Individual A-site elements electronegativity
    chi_Ap = get_electronegativity(A_prime) if A_prime != '-' else 0
    chi_App = get_electronegativity(A_double_prime) if A_double_prime != '-' else 0
    descriptors['chi_A_prime'] = chi_Ap if chi_Ap is not None else 0
    descriptors['chi_A_double_prime'] = chi_App if chi_App is not None else 0
    
    # Individual B-site elements electronegativity
    chi_Bp = get_electronegativity(B_prime) if B_prime != '-' else 0
    chi_Bpp = get_electronegativity(B_double_prime) if B_double_prime != '-' else 0
    chi_Bppp = get_electronegativity(B_triple_prime) if B_triple_prime != '-' else 0
    
    descriptors['chi_B_prime'] = chi_Bp if chi_Bp is not None else 0
    descriptors['chi_B_double_prime'] = chi_Bpp if chi_Bpp is not None else 0
    descriptors['chi_B_triple_prime'] = chi_Bppp if chi_Bppp is not None else 0
    
    # Composition descriptors
    descriptors['x_A_substitution'] = x
    descriptors['y_B_substitution'] = y
    descriptors['z_B_site_occupancy'] = z
    descriptors['alpha_B_triple'] = alpha
    descriptors['oxygen_vacancy'] = calculate_oxygen_vacancy(y, alpha)
    
    # Structural descriptors (from crystal structure)
    descriptors['V_ox'] = V_ox if not np.isnan(V_ox) else 0
    descriptors['Vpseud'] = Vpseud if not np.isnan(Vpseud) else 0
    descriptors['apseud'] = apseud if not np.isnan(apseud) else 0
    
    # Volume-derived descriptors
    if not np.isnan(V_ox) and V_ox > 0:
        descriptors['cubic_root_V'] = V_ox ** (1/3)
    else:
        descriptors['cubic_root_V'] = 0
    
    # Polarizability descriptors
    pol_A, pol_B, pol_avg = calculate_average_polarizability(
        A_prime, A_double_prime, B_prime, B_double_prime, B_triple_prime, x, y, alpha
    )
    descriptors['polarizability_A'] = pol_A
    descriptors['polarizability_B'] = pol_B
    descriptors['polarizability_avg'] = pol_avg
    
    # Mass descriptors (average atomic mass)
    mass_Ap = A_SITE_PROPERTIES.get(A_prime, {}).get('mass', 0) if A_prime != '-' else 0
    mass_App = A_SITE_PROPERTIES.get(A_double_prime, {}).get('mass', 0) if A_double_prime != '-' else 0
    mass_A_avg = mass_Ap * (1 - x) + mass_App * x
    
    mass_Bp = B_SITE_PROPERTIES.get(B_prime, {}).get('mass', 0) if B_prime != '-' else 0
    mass_Bpp = B_SITE_PROPERTIES.get(B_double_prime, {}).get('mass', 0) if B_double_prime != '-' else 0
    mass_Bppp = B_SITE_PROPERTIES.get(B_triple_prime, {}).get('mass', 0) if B_triple_prime != '-' else 0
    
    weight_Bp = max(0, 1 - y - alpha)
    total_weight_B = weight_Bp + y + alpha
    if total_weight_B > 0:
        mass_B_avg = (weight_Bp * mass_Bp + y * mass_Bpp + alpha * mass_Bppp) / total_weight_B
    else:
        mass_B_avg = 0
    
    descriptors['mass_A_avg'] = mass_A_avg
    descriptors['mass_B_avg'] = mass_B_avg
    descriptors['mass_ratio_AB'] = mass_B_avg / mass_A_avg if mass_A_avg > 0 else 0
    
    # =========================================================
    # FIX: Thermal expansion descriptors with physical constraints
    # =========================================================
    alphaLT = row.get('αLT', np.nan) if pd.notna(row.get('αLT', np.nan)) else np.nan
    alphaHT = row.get('αHT', np.nan) if pd.notna(row.get('αHT', np.nan)) else np.nan
    alphaav = row.get('αav', np.nan) if pd.notna(row.get('αav', np.nan)) else np.nan
    
    descriptors['alphaLT'] = alphaLT if not np.isnan(alphaLT) and alphaLT > 0 else np.nan
    descriptors['alphaHT'] = alphaHT if not np.isnan(alphaHT) and alphaHT > 0 else np.nan
    descriptors['alphaav'] = alphaav if not np.isnan(alphaav) and alphaav > 0 else np.nan
    
    # Физическое ограничение: αHT/αLT не может быть > 3 для ферритов
    if not np.isnan(alphaLT) and not np.isnan(alphaHT) and alphaLT > 1e-6:
        ratio = alphaHT / alphaLT
        if ratio > 3.0:
            ratio = np.nan  # Отбрасываем нефизичные значения
        descriptors['alpha_ratio_HT_LT'] = ratio
        descriptors['alpha_diff_HT_LT'] = alphaHT - alphaLT
    else:
        descriptors['alpha_ratio_HT_LT'] = np.nan
        descriptors['alpha_diff_HT_LT'] = np.nan
    
    # =========================================================
    # FIX: ASR descriptors - только положительные значения
    # =========================================================
    asr_600 = row.get('ASR, 600 °C', np.nan)
    if pd.notna(asr_600) and asr_600 > 0 and asr_600 < 100:
        descriptors['ASR_600'] = asr_600
    else:
        descriptors['ASR_600'] = np.nan
    
    asr_650 = row.get('ASR, 650 °C', np.nan)
    if pd.notna(asr_650) and asr_650 > 0 and asr_650 < 100:
        descriptors['ASR_650'] = asr_650
    else:
        descriptors['ASR_650'] = np.nan
    
    asr_700 = row.get('ASR, 700 °C', np.nan)
    if pd.notna(asr_700) and asr_700 > 0 and asr_700 < 100:
        descriptors['ASR_700'] = asr_700
    else:
        descriptors['ASR_700'] = np.nan
    
    # Conductivity descriptors (только положительные значения)
    sigma_500 = row.get('σ (500 °C)', np.nan)
    descriptors['sigma_500'] = sigma_500 if pd.notna(sigma_500) and sigma_500 >= 0 else np.nan
    
    sigma_600 = row.get('σ (600 °C)', np.nan)
    descriptors['sigma_600'] = sigma_600 if pd.notna(sigma_600) and sigma_600 >= 0 else np.nan
    
    sigma_700 = row.get('σ (700 °C)', np.nan)
    descriptors['sigma_700'] = sigma_700 if pd.notna(sigma_700) and sigma_700 >= 0 else np.nan
    
    sigma_max = row.get('σmax', np.nan)
    descriptors['sigma_max'] = sigma_max if pd.notna(sigma_max) and sigma_max >= 0 else np.nan
    
    # Power descriptors
    p_600 = row.get('P(FC), 600 °C', np.nan)
    descriptors['P_600'] = p_600 if pd.notna(p_600) and p_600 >= 0 else np.nan
    
    p_650 = row.get('P(FC), 650 °C', np.nan)
    descriptors['P_650'] = p_650 if pd.notna(p_650) and p_650 >= 0 else np.nan
    
    p_700 = row.get('P(FC), 700 °C', np.nan)
    descriptors['P_700'] = p_700 if pd.notna(p_700) and p_700 >= 0 else np.nan
    
    # Categorical descriptors for encoding
    descriptors['A_prime'] = str(A_prime) if A_prime != '-' else 'Ba'
    descriptors['A_double_prime'] = str(A_double_prime) if A_double_prime != '-' else 'none'
    descriptors['B_prime'] = str(B_prime) if B_prime != '-' else 'Fe'
    descriptors['B_double_prime'] = str(B_double_prime) if B_double_prime != '-' else 'none'
    descriptors['B_triple_prime'] = str(B_triple_prime) if B_triple_prime != '-' else 'none'
    descriptors['doi'] = row.get('doi', '') if pd.notna(row.get('doi', '')) else ''
    
    return descriptors

# =============================================================================
# Train multi-target prediction models (OPTIMIZED VERSION)
# =============================================================================
@st.cache_resource
def train_prediction_models(df_features, fast_mode=True):
    """Train ensemble models for all target properties (OPTIMIZED)"""
    
    # Define feature columns for ML
    feature_cols = [
        'chi_A', 'chi_B', 'chi_diff', 'chi_product',
        'chi_A_prime', 'chi_A_double_prime',
        'chi_B_prime', 'chi_B_double_prime', 'chi_B_triple_prime',
        'x_A_substitution', 'y_B_substitution', 'z_B_site_occupancy',
        'alpha_B_triple', 'oxygen_vacancy',
        'V_ox', 'Vpseud', 'apseud', 'cubic_root_V',
        'polarizability_A', 'polarizability_B', 'polarizability_avg',
        'mass_A_avg', 'mass_B_avg', 'mass_ratio_AB',
        'alphaLT', 'alphaHT', 'alphaav', 'alpha_ratio_HT_LT', 'alpha_diff_HT_LT'
    ]
    
    # Target variables
    target_cols = {
        'sigma_600': 'σ (600 °C)',
        'sigma_max': 'σmax',
        'ASR_600': 'ASR, 600 °C',
        'P_600': 'P(FC), 600 °C',
        'alpha_ratio_HT_LT': 'α_HT/α_LT'
    }
    
    # Prepare data - удаляем строки с NaN в целевых переменных
    valid_indices = []
    for idx, row in df_features.iterrows():
        has_target = False
        for target_key in target_cols.keys():
            val = row.get(target_key, np.nan)
            if pd.notna(val) and val > 0:
                has_target = True
                break
        if has_target:
            valid_indices.append(idx)
    
    if len(valid_indices) < 10:
        return None, df_features
    
    df_valid = df_features.iloc[valid_indices].copy()
    
    # Prepare feature matrix (ONCE, outside the loop)
    available_features = [f for f in feature_cols if f in df_valid.columns]
    X_num = df_valid[available_features].fillna(0)
    
    # Encode categorical variables (ONCE)
    le_Ap = LabelEncoder()
    le_App = LabelEncoder()
    le_Bp = LabelEncoder()
    le_Bpp = LabelEncoder()
    le_Bppp = LabelEncoder()
    
    try:
        X_cat = pd.DataFrame({
            'A_prime_enc': le_Ap.fit_transform(df_valid['A_prime'].astype(str)),
            'A_double_prime_enc': le_App.fit_transform(df_valid['A_double_prime'].astype(str)),
            'B_prime_enc': le_Bp.fit_transform(df_valid['B_prime'].astype(str)),
            'B_double_prime_enc': le_Bpp.fit_transform(df_valid['B_double_prime'].astype(str)),
            'B_triple_prime_enc': le_Bppp.fit_transform(df_valid['B_triple_prime'].astype(str))
        })
    except Exception as e:
        st.warning(f"Error encoding categories: {str(e)}")
        return None, df_valid
    
    X = pd.concat([X_num, X_cat], axis=1)
    feature_names = X.columns.tolist()
    
    # Scale features (ONCE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set hyperparameters based on mode
    if fast_mode:
        n_estimators_xgb = 50
        max_depth_xgb = 3
        n_estimators_rf = 50
        max_depth_rf = 5
    else:
        n_estimators_xgb = 100
        max_depth_xgb = 4
        n_estimators_rf = 100
        max_depth_rf = 6
    
    # Train models for each target
    models = {}
    cv_scores = {}
    
    # Create progress bar for training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    target_items = list(target_cols.items())
    total_targets = len(target_items)
    
    for i, (target_key, target_name) in enumerate(target_items):
        status_text.text(f"Training model for {target_name}... ({i+1}/{total_targets})")
        
        if target_key not in df_valid.columns:
            continue
        
        y_clean = df_valid[target_key].dropna()
        if len(y_clean) < 5:
            continue
        
        y_indices = y_clean.index
        valid_positions = [df_valid.index.get_loc(idx) for idx in y_indices]
        X_scaled_sync = X_scaled[valid_positions]
        y = y_clean.values
        
        # Skip if too few unique values or too few samples (используем numpy/pandas методы)
        if len(np.unique(y)) < 5 or len(y) < 10:
            st.warning(f"Skipping {target_name}: insufficient data ({len(y)} samples, {len(np.unique(y))} unique values)")
            continue
        
        # XGBoost model (primary)
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators_xgb,
            max_depth=max_depth_xgb,
            learning_rate=0.12,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(X_scaled_sync, y)
        
        # Random Forest model (secondary, lighter)
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators_rf,
            max_depth=max_depth_rf,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_scaled_sync, y)
        
        models[target_key] = {
            'xgb': xgb_model,
            'rf': rf_model
        }
        
        # Optional: Quick CV score (only for XGBoost, with fewer folds)
        try:
            from sklearn.model_selection import cross_val_score
            if len(X_scaled) >= 15:
                cv_scores[target_key] = cross_val_score(xgb_model, X_scaled, y, cv=3, scoring='r2').mean()
            else:
                cv_scores[target_key] = 0.0
        except Exception as e:
            cv_scores[target_key] = 0.0
        
        progress_bar.progress((i + 1) / total_targets)
    
    status_text.text("Training complete!")
    progress_bar.empty()
    status_text.empty()
    
    return {
        'models': models,
        'scaler': scaler,
        'feature_names': feature_names,
        'cv_scores': cv_scores,
        'le_Ap': le_Ap,
        'le_App': le_App,
        'le_Bp': le_Bp,
        'le_Bpp': le_Bpp,
        'le_Bppp': le_Bppp,
        'X_scaled': X_scaled,
        'X_df': X,
        'df_features': df_valid,
        'target_cols': target_cols,
        'fast_mode': fast_mode
    }, df_valid

# =============================================================================
# Bubble chart with heatmap overlay (FIXED)
# =============================================================================
def create_bubble_heatmap(df, x_col, y_col, size_col, color_col, title):
    """Create a bubble chart with heatmap overlay for scientific publication"""
    
    # Clean data - drop NaN values and ensure correct dimensions
    clean_df = df[[x_col, y_col, size_col, color_col]].dropna()
    
    if len(clean_df) < 3:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Insufficient data for {title}\nNeed at least 3 points', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    x_vals = clean_df[x_col].values
    y_vals = clean_df[y_col].values
    size_vals = clean_df[size_col].values
    color_vals = clean_df[color_col].values
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create 2D histogram for heatmap background (with error handling)
    try:
        if len(x_vals) >= 10 and len(y_vals) >= 10:
            # Ensure arrays are 1D and have same length
            x_vals_clean = x_vals[~np.isnan(x_vals)]
            y_vals_clean = y_vals[~np.isnan(y_vals)]
            
            # Trim to same length if needed
            min_len = min(len(x_vals_clean), len(y_vals_clean))
            if min_len >= 10:
                x_vals_clean = x_vals_clean[:min_len]
                y_vals_clean = y_vals_clean[:min_len]
                
                hb = ax.hexbin(x_vals_clean, y_vals_clean, gridsize=20, 
                              cmap='Blues', alpha=0.3, mincnt=1)
                plt.colorbar(hb, ax=ax, label='Data density')
    except Exception as e:
        # Silently continue without hexbin
        pass
    
    # Normalize size for bubbles
    if size_vals.max() > size_vals.min():
        sizes = 50 + 200 * (size_vals - size_vals.min()) / (size_vals.max() - size_vals.min())
    else:
        sizes = np.full_like(size_vals, 100)
    
    # Scatter plot with bubbles
    scatter = ax.scatter(x_vals, y_vals, s=sizes, c=color_vals, 
                        cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, ax=ax, label=color_col.replace('_', ' ').title())
    
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line if enough points
    if len(x_vals) >= 4:
        try:
            z = np.polyfit(x_vals, y_vals, 1)
            x_trend = np.linspace(x_vals.min(), x_vals.max(), 50)
            ax.plot(x_trend, np.polyval(z, x_trend), 'r--', alpha=0.5, linewidth=1.5,
                   label=f'Trend: y = {z[0]:.2f}x + {z[1]:.1f}')
            ax.legend(loc='best', fontsize=9)
        except Exception:
            pass
    
    plt.tight_layout()
    return fig

# =============================================================================
# Create property maps for composition space (FULLY REWRITTEN WITH FIXES)
# =============================================================================
def create_property_map(df, x_param, y_param, z_param, title):
    """Create 2D contour map of property across composition space"""
    
    # Clean data - drop NaN values
    clean_df = df[[x_param, y_param, z_param]].dropna()
    
    if len(clean_df) < 4:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Insufficient data for {title}\nNeed at least 4 points', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    x_vals = clean_df[x_param].values
    y_vals = clean_df[y_param].values
    z_vals = clean_df[z_param].values
    
    # Ensure arrays are 1D
    x_vals = x_vals.flatten()
    y_vals = y_vals.flatten()
    z_vals = z_vals.flatten()
    
    # =========================================================
    # FIX: Синхронизация длин массивов (главное исправление)
    # =========================================================
    min_len = min(len(x_vals), len(y_vals), len(z_vals))
    x_vals = x_vals[:min_len]
    y_vals = y_vals[:min_len]
    z_vals = z_vals[:min_len]
    
    # Remove any remaining NaN
    valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isnan(z_vals))
    x_vals = x_vals[valid_mask]
    y_vals = y_vals[valid_mask]
    z_vals = z_vals[valid_mask]
    
    # =========================================================
    # FIX: Физические ограничения для α_HT/α_LT
    # =========================================================
    if z_param == 'alpha_ratio_HT_LT':
        # Удаляем нефизичные точки (отношение > 3 или < 0.5)
        valid_ratio_mask = (z_vals <= 3.0) & (z_vals >= 0.5)
        x_vals = x_vals[valid_ratio_mask]
        y_vals = y_vals[valid_ratio_mask]
        z_vals = z_vals[valid_ratio_mask]
        
        if len(x_vals) < 4:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(x_vals, y_vals, c=z_vals, cmap='RdYlBu_r', 
                      s=80, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='α_HT/α_LT')
            ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)
            ax.set_title(title + ' (scatter plot, filtered data)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
    
    # =========================================================
    # FIX: Физические ограничения для ASR (только положительные)
    # =========================================================
    if z_param in ['ASR_600', 'ASR_650', 'ASR_700']:
        # Удаляем отрицательные или нулевые значения
        valid_asr_mask = z_vals > 0
        x_vals = x_vals[valid_asr_mask]
        y_vals = y_vals[valid_asr_mask]
        z_vals = z_vals[valid_asr_mask]
        
        if len(x_vals) < 4:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(x_vals, y_vals, c=z_vals, cmap='RdYlBu_r', 
                      s=80, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label=z_param.replace('_', ' ').title())
            ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)
            ax.set_title(title + ' (scatter plot, positive values only)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
    
    if len(x_vals) < 4:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Insufficient valid data points ({len(x_vals)})\nNeed at least 4', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    # Create grid
    try:
        x_grid = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_grid = np.linspace(y_vals.min(), y_vals.max(), 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate with fallback methods
        try:
            Z = griddata((x_vals, y_vals), z_vals, (X, Y), method='cubic')
            if np.isnan(Z).all():
                Z = griddata((x_vals, y_vals), z_vals, (X, Y), method='linear')
        except Exception:
            Z = griddata((x_vals, y_vals), z_vals, (X, Y), method='linear')
        
        # If still all NaN, create simple scatter
        if np.isnan(Z).all():
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(x_vals, y_vals, c=z_vals, cmap='RdYlBu_r', 
                                s=80, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label=z_param.replace('_', ' ').title())
            ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)
            ax.set_title(title + ' (scatter plot)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # =========================================================
        # FIX: Ограничение цветовой шкалы для физических величин
        # =========================================================
        if z_param in ['ASR_600', 'ASR_650', 'ASR_700']:
            Z = np.maximum(Z, 0)  # Обрезаем отрицательные значения
            vmin, vmax = 0, min(10, np.nanmax(Z))  # ASR обычно < 10
            contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.8, vmin=vmin, vmax=vmax)
        elif z_param == 'alpha_ratio_HT_LT':
            Z = np.clip(Z, 0.5, 3.0)  # Ограничиваем физическим диапазоном
            contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.8, vmin=0.5, vmax=3.0)
        elif z_param in ['sigma_600', 'sigma_max', 'sigma_500', 'sigma_700']:
            Z = np.maximum(Z, 0)  # Проводимость не может быть отрицательной
            contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.8, vmin=0)
        else:
            contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
        
        plt.colorbar(contour, ax=ax, label=z_param.replace('_', ' ').title())
        
        # Add contour lines
        contour_lines = ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.3)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        # Add data points
        ax.scatter(x_vals, y_vals, c='black', s=30, alpha=0.5, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Error creating map: {str(e)[:100]}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

# =============================================================================
# Parallel coordinates for multi-dimensional analysis
# =============================================================================
def create_parallel_coordinates(df, features, color_by):
    """Create parallel coordinates plot for multi-dimensional analysis"""
    
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        return None
    
    # Normalize features
    df_norm = df[available_features].copy()
    for col in available_features:
        if col != color_by:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    if color_by in df.columns:
        df_norm[color_by] = df[color_by]
    
    fig = px.parallel_coordinates(
        df_norm,
        color=color_by if color_by in df_norm.columns else None,
        color_continuous_scale='RdYlBu_r',
        title='Multi-dimensional Property Analysis',
        labels={col: col.replace('_', ' ').title() for col in available_features}
    )
    
    fig.update_layout(width=1000, height=600)
    return fig

# =============================================================================
# Correlation matrix with hierarchical clustering
# =============================================================================
def create_correlation_heatmap(df, features):
    """Create correlation heatmap with hierarchical clustering"""
    
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        return None
    
    corr_matrix = df[available_features].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
               cmap='RdBu_r', center=0, square=True, ax=ax,
               cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
               annot_kws={'size': 8})
    
    ax.set_title('Property Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    return fig

# =============================================================================
# Clustering analysis
# =============================================================================
def perform_clustering_analysis(df, features, n_clusters=4):
    """Perform K-means clustering on materials"""
    
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2 or len(df) < n_clusters:
        return None, None, None
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    X = df[available_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10',
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_title(f'Material Clustering (k={n_clusters})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, clusters, kmeans

# =============================================================================
# Predict properties for user-defined composition
# =============================================================================
def predict_composition(model_data, A_prime, A_double_prime, B_prime, B_double_prime,
                        B_triple_prime, B_quad_prime, x, y, z, alpha):
    """Predict all properties for a given composition"""
    
    if model_data is None:
        return None
    
    # Create input row
    input_row = pd.Series({
        "A'": A_prime,
        "A''": A_double_prime,
        "B'": B_prime,
        "B''": B_double_prime,
        "B'''": B_triple_prime,
        "B''''": B_quad_prime,
        'x': x,
        'y': y,
        'z': z,
        'α': alpha
    })
    
    # Calculate descriptors
    desc = calculate_descriptors(input_row)
    
    # Prepare feature vector
    feature_names = model_data['feature_names']
    
    X_pred = pd.DataFrame([{
        'chi_A': desc.get('chi_A', 0),
        'chi_B': desc.get('chi_B', 0),
        'chi_diff': desc.get('chi_diff', 0),
        'chi_product': desc.get('chi_product', 0),
        'chi_A_prime': desc.get('chi_A_prime', 0),
        'chi_A_double_prime': desc.get('chi_A_double_prime', 0),
        'chi_B_prime': desc.get('chi_B_prime', 0),
        'chi_B_double_prime': desc.get('chi_B_double_prime', 0),
        'chi_B_triple_prime': desc.get('chi_B_triple_prime', 0),
        'x_A_substitution': desc.get('x_A_substitution', x),
        'y_B_substitution': desc.get('y_B_substitution', y),
        'z_B_site_occupancy': desc.get('z_B_site_occupancy', z),
        'alpha_B_triple': desc.get('alpha_B_triple', alpha),
        'oxygen_vacancy': desc.get('oxygen_vacancy', 0),
        'V_ox': desc.get('V_ox', 0),
        'Vpseud': desc.get('Vpseud', 0),
        'apseud': desc.get('apseud', 0),
        'cubic_root_V': desc.get('cubic_root_V', 0),
        'polarizability_A': desc.get('polarizability_A', 0),
        'polarizability_B': desc.get('polarizability_B', 0),
        'polarizability_avg': desc.get('polarizability_avg', 0),
        'mass_A_avg': desc.get('mass_A_avg', 0),
        'mass_B_avg': desc.get('mass_B_avg', 0),
        'mass_ratio_AB': desc.get('mass_ratio_AB', 0),
        'alphaLT': desc.get('alphaLT', 0),
        'alphaHT': desc.get('alphaHT', 0),
        'alphaav': desc.get('alphaav', 0),
        'alpha_ratio_HT_LT': desc.get('alpha_ratio_HT_LT', 0),
        'alpha_diff_HT_LT': desc.get('alpha_diff_HT_LT', 0)
    }])
    
    # Add encoded categorical features
    try:
        X_pred['A_prime_enc'] = model_data['le_Ap'].transform([str(A_prime)])[0]
    except:
        X_pred['A_prime_enc'] = -1
    
    try:
        X_pred['A_double_prime_enc'] = model_data['le_App'].transform([str(A_double_prime)])[0]
    except:
        X_pred['A_double_prime_enc'] = -1
    
    try:
        X_pred['B_prime_enc'] = model_data['le_Bp'].transform([str(B_prime)])[0]
    except:
        X_pred['B_prime_enc'] = -1
    
    try:
        X_pred['B_double_prime_enc'] = model_data['le_Bpp'].transform([str(B_double_prime)])[0]
    except:
        X_pred['B_double_prime_enc'] = -1
    
    try:
        X_pred['B_triple_prime_enc'] = model_data['le_Bppp'].transform([str(B_triple_prime)])[0]
    except:
        X_pred['B_triple_prime_enc'] = -1
    
    # Ensure all features present
    for col in feature_names:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    X_pred = X_pred[feature_names]
    X_pred_scaled = model_data['scaler'].transform(X_pred)
    
    # Make predictions for each target
    predictions = {}
    
    for target_key, target_models in model_data['models'].items():
        pred_xgb = target_models['xgb'].predict(X_pred_scaled)[0]
        pred_rf = target_models['rf'].predict(X_pred_scaled)[0]
        
        # Ensemble with weights (XGBoost 0.6, RF 0.4)
        pred_ensemble = 0.6 * pred_xgb + 0.4 * pred_rf
        
        # =========================================================
        # FIX: Физические ограничения для предсказаний
        # =========================================================
        if target_key in ['ASR_600', 'ASR_650', 'ASR_700']:
            pred_ensemble = max(0, pred_ensemble)
            pred_xgb = max(0, pred_xgb)
            pred_rf = max(0, pred_rf)
        elif target_key == 'alpha_ratio_HT_LT':
            pred_ensemble = np.clip(pred_ensemble, 0.5, 3.0)
            pred_xgb = np.clip(pred_xgb, 0.5, 3.0)
            pred_rf = np.clip(pred_rf, 0.5, 3.0)
        elif target_key in ['sigma_600', 'sigma_max', 'P_600']:
            pred_ensemble = max(0, pred_ensemble)
            pred_xgb = max(0, pred_xgb)
            pred_rf = max(0, pred_rf)
        
        predictions[target_key] = {
            'ensemble': pred_ensemble,
            'xgb': pred_xgb,
            'rf': pred_rf
        }
    
    return predictions

# =============================================================================
# Modern UI Components (SAME AS ORIGINAL)
# =============================================================================
def apply_modern_styling():
    """Apply modern CSS styling to the app"""
    st.markdown("""
    <style>
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --background-color: #f8f9fa;
        --text-color: #212529;
        --border-color: #dee2e6;
    }
    
    .main {
        background-color: var(--background-color);
        padding: 2rem;
    }
    
    h1 {
        color: var(--primary-color);
        font-weight: 600;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: var(--text-color);
        font-weight: 500;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: var(--text-color);
        font-weight: 500;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--primary-color), #45a049);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: white;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    .stButton button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .css-1d391kg {
        background-color: white;
        border-right: 1px solid var(--border-color);
    }
    
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-weight: 500;
    }
    
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .dataframe th {
        background-color: var(--primary-color);
        color: white;
        font-weight: 500;
        padding: 0.75rem !important;
    }
    
    .dataframe td {
        padding: 0.5rem !important;
        border-bottom: 1px solid var(--border-color);
    }
    
    .stProgress > div > div {
        background-color: var(--primary-color);
        border-radius: 10px;
    }
    
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .stSlider div[data-baseweb="slider"] {
        margin-top: 0.5rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }
    
    .version-badge {
        display: inline-block;
        background-color: var(--primary-color);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-left: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# Progress bar context manager (SAME AS ORIGINAL)
# =============================================================================
class ModernProgressBar:
    def __init__(self, message, total_steps, show_time=True):
        self.message = message
        self.total_steps = total_steps
        self.progress_bar = None
        self.status_text = None
        self.time_text = None
        self.show_time = show_time
        self.start_time = None
    
    def __enter__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        if self.show_time:
            self.time_text = st.empty()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.empty()
        self.status_text.empty()
        if self.time_text:
            self.time_text.empty()
    
    def update(self, step, sub_message=""):
        progress = step / self.total_steps
        self.progress_bar.progress(progress)
        
        elapsed = time.time() - self.start_time
        if step > 0:
            eta = (elapsed / step) * (self.total_steps - step)
            time_str = f" | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
        else:
            time_str = ""
        
        self.status_text.text(f"{self.message}: {sub_message} ({int(progress*100)}%)")
        if self.show_time and self.time_text:
            self.time_text.text(f"⏱️{time_str}")

# =============================================================================
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(
        page_title="BaFeO₃ Multi-Property Predictor v1.2",
        page_icon="🧲",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply modern styling
    apply_modern_styling()
    
    # Title
    st.markdown("""
    <h1>
        🧲 BaFeO₃ Multi-Property Predictor for Doped Ferrites
        <span class="version-badge">v1.2</span>
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p style="font-size: 1.1rem; margin-bottom: 0;">
        Advanced computational platform for analyzing and predicting electrical conductivity,
        area-specific resistance (ASR), power output, and thermal expansion of doped BaFeO₃-based
        perovskites for solid oxide fuel cell applications. Uses electronegativity-based descriptors.
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #2ca02c;">
        ✅ v1.2: Fixed NaN handling, physical constraints for α_HT/α_LT (≤3.0), ASR (≥0), and robust interpolation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # File Upload Section
    # =========================================================================
    st.markdown("## 📂 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Excel file with ferrite compositions and properties",
        type=['xlsx', 'xls'],
        help="File must contain columns: no. paper, Composition, A', A'', B', B'', B''', B'''\"', x, y, z, α, etc."
    )
    
    # Fast mode toggle in sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        fast_mode = st.checkbox("Fast mode (faster training, slightly lower accuracy)", value=True)
    
    if uploaded_file is not None:
        with ModernProgressBar("Loading and validating data", 3) as pb:
            pb.update(1, "Reading Excel file")
            df_raw = load_uploaded_excel(uploaded_file)
            
            if df_raw is None:
                st.error("Failed to load file. Please check format.")
                return
            
            pb.update(2, "Calculating descriptors")
            # Calculate descriptors for all rows
            descriptor_list = []
            for idx, row in df_raw.iterrows():
                try:
                    desc = calculate_descriptors(row)
                    descriptor_list.append(desc)
                except Exception as e:
                    st.warning(f"Error processing row {idx}: {str(e)}")
                    continue
            
            df_descriptors = pd.DataFrame(descriptor_list)
            pb.update(3, "Training ML models")
            
            # Train models with fast mode option
            model_data, df_features = train_prediction_models(df_descriptors, fast_mode=fast_mode)
            
            if model_data is None:
                st.warning("Insufficient data for training models. Need at least 10 valid samples.")
                return
            
            st.success(f"✅ Loaded {len(df_raw)} compositions, {len(df_features)} with valid properties")
            if fast_mode:
                st.info("⚡ Fast mode enabled: using XGBoost (n=50) + RF (n=50) without CV")
            else:
                st.info("🎯 Accurate mode: using XGBoost (n=100) + RF (n=100)")
    else:
        st.info("👈 Please upload an Excel file to begin analysis")
        st.markdown("""
        <div class="card">
            <h4>Expected file format:</h4>
            <p>The Excel file should contain the following columns:</p>
            <ul>
                <li><b>Composition identifiers:</b> no. paper, Composition</li>
                <li><b>A-site:</b> A', A'', x</li>
                <li><b>B-site:</b> B', B'', B''', B'''", y, z, α</li>
                <li><b>Structural:</b> a(ox), b(ox), c(ox), V(ox), Vpseud(ox), apseud(ox)</li>
                <li><b>Electrical:</b> σ (500 °C), σ (600 °C), σ (700 °C), σmax</li>
                <li><b>Thermal:</b> αLT, αHT, αav</li>
                <li><b>Electrochemical:</b> P(FC), ASR at 600, 650, 700 °C</li>
                <li><b>Reference:</b> doi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # =========================================================================
    # Sidebar Navigation
    # =========================================================================
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        page = st.radio(
            "Select Module",
            ["📊 Data Explorer",
             "🔬 Property Maps & Optimization",
             "🎯 Bubble Charts & Heatmaps",
             "🤖 ML Predictor",
             "📈 Multi-dimensional Analysis",
             "🌲 Clustering Analysis",
             "📊 Correlation Analysis",
             "ℹ️ About"]
        )
        
        st.markdown("---")
        
        # Database stats
        st.markdown("## 📊 Database Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Compositions</div>
                <div class="metric-value">{len(df_features)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_A = df_features['A_prime'].nunique() if 'A_prime' in df_features.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">A-site Elements</div>
                <div class="metric-value">{unique_A}</div>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            unique_B = df_features['B_prime'].nunique() if 'B_prime' in df_features.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">B-site Elements</div>
                <div class="metric-value">{unique_B}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            valid_sigma = df_features['sigma_600'].notna().sum() if 'sigma_600' in df_features.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">σ(600°C) values</div>
                <div class="metric-value">{valid_sigma}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model performance if available
        if model_data and model_data['cv_scores']:
            st.markdown("## 📈 Model Performance (R²)")
            for target, score in model_data['cv_scores'].items():
                if score > 0:
                    target_name = model_data['target_cols'].get(target, target)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{target_name}</div>
                        <div class="metric-value">{score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # =========================================================================
    # Page 1: Data Explorer
    # =========================================================================
    if page == "📊 Data Explorer":
        st.markdown("## 📊 Ferrite Database Explorer")
        
        # Filters
        with st.expander("🔍 Filter Data", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'A_prime' in df_features.columns:
                    a_options = ['All'] + sorted(df_features['A_prime'].unique().tolist())
                    selected_a = st.selectbox("A-site (primary)", a_options)
                else:
                    selected_a = 'All'
            
            with col2:
                if 'B_prime' in df_features.columns:
                    b_options = ['All'] + sorted(df_features['B_prime'].unique().tolist())
                    selected_b = st.selectbox("B-site (primary)", b_options)
                else:
                    selected_b = 'All'
            
            with col3:
                sigma_min = st.slider("Min σ(600°C) (S/cm)", 0.0, 200.0, 0.0)
        
        # Apply filters
        filtered_df = df_features.copy()
        if selected_a != 'All' and 'A_prime' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['A_prime'] == selected_a]
        if selected_b != 'All' and 'B_prime' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['B_prime'] == selected_b]
        if sigma_min > 0 and 'sigma_600' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sigma_600'] >= sigma_min]
        
        # Statistics
        st.markdown("### 📈 Dataset Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Filtered Compositions", len(filtered_df))
        
        with col2:
            if 'sigma_600' in filtered_df.columns:
                mean_sigma = filtered_df['sigma_600'].mean()
                st.metric("σ(600°C) mean", f"{mean_sigma:.1f}" if not pd.isna(mean_sigma) else "N/A", "S/cm")
        
        with col3:
            if 'ASR_600' in filtered_df.columns:
                mean_asr = filtered_df['ASR_600'].mean()
                st.metric("ASR(600°C) mean", f"{mean_asr:.2f}" if not pd.isna(mean_asr) else "N/A", "Ω·cm²")
        
        with col4:
            if 'alpha_ratio_HT_LT' in filtered_df.columns:
                mean_ratio = filtered_df['alpha_ratio_HT_LT'].mean()
                st.metric("α_HT/α_LT mean", f"{mean_ratio:.2f}" if not pd.isna(mean_ratio) else "N/A")
        
        with col5:
            if 'oxygen_vacancy' in filtered_df.columns:
                st.metric("Oxygen Vacancy", f"{filtered_df['oxygen_vacancy'].mean():.2f}")
        
        # Data table
        st.markdown("### 📋 Composition Data")
        
        display_cols = ['A_prime', 'A_double_prime', 'B_prime', 'B_double_prime', 
                       'x_A_substitution', 'y_B_substitution', 'sigma_600', 'ASR_600']
        available_display = [c for c in display_cols if c in filtered_df.columns]
        
        if available_display:
            st.dataframe(filtered_df[available_display].head(50), use_container_width=True)
        else:
            st.dataframe(filtered_df.head(50), use_container_width=True)
        
        # Distribution plots
        st.markdown("### 📊 Property Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'sigma_600' in filtered_df.columns:
                sigma_clean = filtered_df['sigma_600'].dropna()
                if len(sigma_clean) > 0:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(sigma_clean, bins=20, 
                           color=MODERN_COLORS['primary'], edgecolor='white', alpha=0.7)
                    ax.set_xlabel('σ(600°C) (S/cm)', fontsize=11)
                    ax.set_ylabel('Frequency', fontsize=11)
                    ax.set_title('Conductivity Distribution', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
        
        with col2:
            if 'ASR_600' in filtered_df.columns:
                asr_clean = filtered_df['ASR_600'].dropna()
                if len(asr_clean) > 0:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(asr_clean, bins=20, 
                           color=MODERN_COLORS['secondary'], edgecolor='white', alpha=0.7)
                    ax.set_xlabel('ASR(600°C) (Ω·cm²)', fontsize=11)
                    ax.set_ylabel('Frequency', fontsize=11)
                    ax.set_title('ASR Distribution', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
    
    # =========================================================================
    # Page 2: Property Maps & Optimization
    # =========================================================================
    elif page == "🔬 Property Maps & Optimization":
        st.markdown("## 🔬 Composition-Property Maps")
        
        st.markdown("""
        <div class="card">
            <p>Explore how properties vary across composition space. Use these maps to identify
            optimal regions for high conductivity, low ASR, and favorable thermal expansion.</p>
            <p><strong>Note:</strong> α_HT/α_LT is physically constrained to 0.5-3.0 range. ASR is constrained to ≥0.</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Conductivity Maps", "ASR Maps", "Thermal Expansion Maps"])
        
        with tab1:
            st.markdown("### Conductivity vs Composition")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_param = st.selectbox("X-axis parameter", 
                                      ['x_A_substitution', 'y_B_substitution', 'chi_diff', 'oxygen_vacancy'],
                                      index=0, key='cond_x')
            
            with col2:
                y_param = st.selectbox("Y-axis parameter",
                                      ['y_B_substitution', 'x_A_substitution', 'chi_diff', 'oxygen_vacancy'],
                                      index=1, key='cond_y')
            
            if 'sigma_600' in df_features.columns:
                fig = create_property_map(df_features, x_param, y_param, 'sigma_600',
                                         f'σ(600°C) vs {x_param} vs {y_param}')
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            if 'sigma_max' in df_features.columns:
                fig = create_property_map(df_features, x_param, y_param, 'sigma_max',
                                         f'σmax vs {x_param} vs {y_param}')
                if fig:
                    st.pyplot(fig)
                    plt.close()
        
        with tab2:
            st.markdown("### ASR vs Composition")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_param = st.selectbox("X-axis parameter",
                                      ['x_A_substitution', 'y_B_substitution', 'chi_diff', 'oxygen_vacancy'],
                                      index=0, key='asr_x')
            
            with col2:
                y_param = st.selectbox("Y-axis parameter",
                                      ['y_B_substitution', 'x_A_substitution', 'chi_diff', 'oxygen_vacancy'],
                                      index=1, key='asr_y')
            
            if 'ASR_600' in df_features.columns:
                fig = create_property_map(df_features, x_param, y_param, 'ASR_600',
                                         f'ASR(600°C) vs {x_param} vs {y_param}')
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            if 'ASR_700' in df_features.columns:
                fig = create_property_map(df_features, x_param, y_param, 'ASR_700',
                                         f'ASR(700°C) vs {x_param} vs {y_param}')
                if fig:
                    st.pyplot(fig)
                    plt.close()
        
        with tab3:
            st.markdown("### Thermal Expansion vs Composition")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_param = st.selectbox("X-axis parameter",
                                      ['x_A_substitution', 'y_B_substitution', 'chi_diff', 'oxygen_vacancy'],
                                      index=0, key='te_x')
            
            with col2:
                y_param = st.selectbox("Y-axis parameter",
                                      ['y_B_substitution', 'x_A_substitution', 'chi_diff', 'oxygen_vacancy'],
                                      index=1, key='te_y')
            
            if 'alpha_ratio_HT_LT' in df_features.columns:
                fig = create_property_map(df_features, x_param, y_param, 'alpha_ratio_HT_LT',
                                         f'α_HT/α_LT vs {x_param} vs {y_param}')
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            if 'alpha_diff_HT_LT' in df_features.columns:
                fig = create_property_map(df_features, x_param, y_param, 'alpha_diff_HT_LT',
                                         f'Δα (HT-LT) vs {x_param} vs {y_param}')
                if fig:
                    st.pyplot(fig)
                    plt.close()
    
    # =========================================================================
    # Page 3: Bubble Charts & Heatmaps
    # =========================================================================
    elif page == "🎯 Bubble Charts & Heatmaps":
        st.markdown("## 🎯 Advanced Bubble Charts with Heatmap Overlays")
        
        st.markdown("""
        <div class="card">
            <p>Bubble charts combine three dimensions: x-axis, y-axis, bubble size, and bubble color.
            Heatmap background shows data density. Ideal for identifying trends and outliers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            x_options = ['chi_diff', 'oxygen_vacancy', 'x_A_substitution', 'y_B_substitution',
                        'polarizability_avg', 'mass_ratio_AB']
            x_col = st.selectbox("X-axis", [o for o in x_options if o in df_features.columns], key='bubble_x')
        
        with col2:
            y_options = ['sigma_600', 'ASR_600', 'sigma_max', 'alpha_ratio_HT_LT']
            y_col = st.selectbox("Y-axis", [o for o in y_options if o in df_features.columns], key='bubble_y')
        
        col1, col2 = st.columns(2)
        
        with col1:
            size_options = ['sigma_600', 'ASR_600', 'sigma_max', 'P_600']
            size_col = st.selectbox("Bubble size", [o for o in size_options if o in df_features.columns], key='bubble_size')
        
        with col2:
            color_options = ['chi_diff', 'oxygen_vacancy', 'y_B_substitution', 'x_A_substitution']
            color_col = st.selectbox("Bubble color", [o for o in color_options if o in df_features.columns], key='bubble_color')
        
        # Create bubble chart
        if x_col in df_features.columns and y_col in df_features.columns:
            fig = create_bubble_heatmap(df_features, x_col, y_col, size_col, color_col,
                                       f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}')
            if fig:
                st.pyplot(fig)
                plt.close()
        
        # Additional bubble charts
        st.markdown("### Additional Bubble Chart Combinations")
        
        bubble_type = st.radio("Select visualization", 
                              ["Conductivity vs Electronegativity", 
                               "ASR vs Oxygen Vacancy",
                               "Power vs Composition",
                               "Thermal Expansion vs Polarizability"],
                              horizontal=True)
        
        if bubble_type == "Conductivity vs Electronegativity":
            if 'sigma_600' in df_features.columns and 'chi_diff' in df_features.columns:
                fig = create_bubble_heatmap(df_features, 'chi_diff', 'sigma_600', 
                                           'sigma_max', 'y_B_substitution',
                                           'Conductivity vs Electronegativity Difference')
                if fig:
                    st.pyplot(fig)
                    plt.close()
        
        elif bubble_type == "ASR vs Oxygen Vacancy":
            if 'ASR_600' in df_features.columns and 'oxygen_vacancy' in df_features.columns:
                fig = create_bubble_heatmap(df_features, 'oxygen_vacancy', 'ASR_600',
                                           'y_B_substitution', 'chi_diff',
                                           'ASR vs Oxygen Vacancy Concentration')
                if fig:
                    st.pyplot(fig)
                    plt.close()
        
        elif bubble_type == "Power vs Composition":
            if 'P_600' in df_features.columns and 'y_B_substitution' in df_features.columns:
                fig = create_bubble_heatmap(df_features, 'y_B_substitution', 'P_600',
                                           'sigma_600', 'x_A_substitution',
                                           'Power Output vs Dopant Concentration')
                if fig:
                    st.pyplot(fig)
                    plt.close()
        
        elif bubble_type == "Thermal Expansion vs Polarizability":
            if 'alpha_ratio_HT_LT' in df_features.columns and 'polarizability_avg' in df_features.columns:
                fig = create_bubble_heatmap(df_features, 'polarizability_avg', 'alpha_ratio_HT_LT',
                                           'y_B_substitution', 'chi_diff',
                                           'Thermal Expansion Ratio vs Polarizability')
                if fig:
                    st.pyplot(fig)
                    plt.close()
    
    # =========================================================================
    # Page 4: ML Predictor
    # =========================================================================
    elif page == "🤖 ML Predictor":
        st.markdown("## 🤖 Composition Property Predictor")
        
        if model_data is None:
            st.warning("Model not trained. Need more data.")
            return
        
        st.markdown("""
        <div class="card">
            <p>Enter the composition of your doped BaFeO₃ perovskite to predict its properties.
            The prediction uses ensemble machine learning models (XGBoost + Random Forest)
            trained on experimental data with electronegativity-based descriptors.</p>
            <p><strong>Physical constraints applied:</strong> ASR ≥ 0, α_HT/α_LT ∈ [0.5, 3.0], σ ≥ 0</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Composition input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**A-site**")
            a_options = list(A_SITE_PROPERTIES.keys())
            A_prime = st.selectbox("A' (primary)", a_options, index=a_options.index('Ba') if 'Ba' in a_options else 0)
            A_double_prime = st.selectbox("A'' (substituent)", ['-'] + a_options, index=0)
            x = st.slider("x (A'' fraction)", 0.0, 1.0, 0.5, 0.01)
        
        with col2:
            st.markdown("**B-site**")
            b_options = list(B_SITE_PROPERTIES.keys())
            B_prime = st.selectbox("B' (primary)", b_options, index=b_options.index('Fe') if 'Fe' in b_options else 0)
            B_double_prime = st.selectbox("B'' (first substituent)", ['-'] + b_options, index=0)
            B_triple_prime = st.selectbox("B''' (second substituent)", ['-'] + b_options, index=0)
            B_quad_prime = st.selectbox("B'''' (third substituent)", ['-'] + b_options, index=0)
        
        with col3:
            st.markdown("**Stoichiometry**")
            y = st.slider("y (B'' fraction)", 0.0, 0.5, 0.1, 0.01)
            alpha = st.slider("α (B''' fraction)", 0.0, 0.5, 0.0, 0.01)
            z = st.slider("z (A-site occupancy)", 0.8, 1.0, 1.0, 0.01)
        
        if st.button("🔮 Predict Properties", use_container_width=True):
            with st.spinner("Calculating descriptors and predicting..."):
                predictions = predict_composition(
                    model_data, A_prime, A_double_prime, B_prime, B_double_prime,
                    B_triple_prime, B_quad_prime, x, y, z, alpha
                )
            
            if predictions:
                st.markdown("### 📊 Predicted Properties")
                
                # Display predictions in cards
                cols = st.columns(3)
                
                pred_items = []
                for target_key, pred_dict in predictions.items():
                    target_name = model_data['target_cols'].get(target_key, target_key)
                    pred_items.append((target_name, pred_dict['ensemble']))
                
                for i, (name, value) in enumerate(pred_items[:3]):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{name}</div>
                            <div class="metric-value">{value:.2f}</div>
                            <div class="metric-delta">Ensemble prediction</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if len(pred_items) > 3:
                    cols2 = st.columns(2)
                    for i, (name, value) in enumerate(pred_items[3:]):
                        with cols2[i % 2]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{name}</div>
                                <div class="metric-value">{value:.2f}</div>
                                <div class="metric-delta">Ensemble prediction</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Model comparison
                with st.expander("📊 Model Comparison Details"):
                    comparison_data = []
                    for target_key, pred_dict in predictions.items():
                        target_name = model_data['target_cols'].get(target_key, target_key)
                        comparison_data.append({
                            'Property': target_name,
                            'XGBoost': f"{pred_dict['xgb']:.2f}",
                            'Random Forest': f"{pred_dict['rf']:.2f}",
                            'Ensemble': f"{pred_dict['ensemble']:.2f}"
                        })
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                
                # Display calculated descriptors
                with st.expander("📐 Calculated Descriptors"):
                    input_row = pd.Series({
                        "A'": A_prime, "A''": A_double_prime,
                        "B'": B_prime, "B''": B_double_prime,
                        "B'''": B_triple_prime, "B''''": B_quad_prime,
                        'x': x, 'y': y, 'z': z, 'α': alpha
                    })
                    desc = calculate_descriptors(input_row)
                    
                    desc_display = {
                        'χ(A)': f"{desc.get('chi_A', 0):.3f}",
                        'χ(B)': f"{desc.get('chi_B', 0):.3f}",
                        '|χ(A)-χ(B)|': f"{desc.get('chi_diff', 0):.3f}",
                        'Oxygen Vacancy': f"{desc.get('oxygen_vacancy', 0):.3f}",
                        'Polarizability (avg)': f"{desc.get('polarizability_avg', 0):.3f}",
                        'Mass Ratio (B/A)': f"{desc.get('mass_ratio_AB', 0):.2f}"
                    }
                    
                    desc_df = pd.DataFrame([desc_display]).T
                    desc_df.columns = ['Value']
                    st.dataframe(desc_df, use_container_width=True)
            else:
                st.error("Prediction failed. Please check input values.")
    
    # =========================================================================
    # Page 5: Multi-dimensional Analysis
    # =========================================================================
    elif page == "📈 Multi-dimensional Analysis":
        st.markdown("## 📈 Multi-dimensional Property Analysis")
        
        st.markdown("""
        <div class="card">
            <p>Parallel coordinates allow visualization of high-dimensional data.
            Each line represents a material, and each axis represents a property.
            Patterns reveal correlations and trade-offs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Select features for parallel coordinates
        all_features = ['sigma_600', 'ASR_600', 'chi_diff', 'oxygen_vacancy', 
                       'x_A_substitution', 'y_B_substitution', 'alpha_ratio_HT_LT',
                       'polarizability_avg', 'mass_ratio_AB']
        
        available_features = [f for f in all_features if f in df_features.columns]
        
        selected_features = st.multiselect(
            "Select properties for parallel coordinates",
            available_features,
            default=available_features[:min(6, len(available_features))]
        )
        
        color_by = st.selectbox("Color by", 
                               ['sigma_600', 'ASR_600', 'chi_diff', 'oxygen_vacancy'],
                               index=0 if 'sigma_600' in df_features.columns else 0)
        
        if len(selected_features) >= 2:
            fig = create_parallel_coordinates(df_features, selected_features, color_by)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Radar charts for material comparison
        st.markdown("### 🎯 Material Comparison Radar Charts")
        
        if len(df_features) >= 3:
            # Select materials to compare
            material_options = []
            for idx, row in df_features.iterrows():
                label = f"{row.get('A_prime', '')}{row.get('B_prime', '')}"
                if 'y_B_substitution' in row:
                    label += f" y={row['y_B_substitution']:.2f}"
                material_options.append(label)
            
            selected_materials = st.multiselect(
                "Select materials to compare",
                list(zip(material_options, df_features.index)),
                format_func=lambda x: x[0],
                max_selections=5
            )
            
            radar_features = ['sigma_600', 'ASR_600', 'chi_diff', 'alpha_ratio_HT_LT', 'oxygen_vacancy']
            radar_available = [f for f in radar_features if f in df_features.columns]
            
            if selected_materials and len(radar_available) >= 3:
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                
                angles = np.linspace(0, 2 * np.pi, len(radar_available), endpoint=False).tolist()
                angles += angles[:1]
                
                for label, idx in selected_materials:
                    values = []
                    for feat in radar_available:
                        val = df_features.loc[idx, feat] if pd.notna(df_features.loc[idx, feat]) else 0
                        # Normalize
                        max_val = df_features[feat].max()
                        if max_val > 0:
                            val = val / max_val
                        values.append(val)
                    values += values[:1]
                    
                    ax.plot(angles, values, 'o-', linewidth=2, label=label)
                    ax.fill(angles, values, alpha=0.25)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([f.replace('_', ' ').title() for f in radar_available], fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_title('Material Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
                ax.grid(True)
                
                st.pyplot(fig)
                plt.close()
    
    # =========================================================================
    # Page 6: Clustering Analysis
    # =========================================================================
    elif page == "🌲 Clustering Analysis":
        st.markdown("## 🌲 Material Clustering Analysis")
        
        st.markdown("""
        <div class="card">
            <p>Clustering groups materials with similar properties. Use this to identify families
            of compositions with characteristic behavior.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Select features for clustering
        cluster_features = ['sigma_600', 'ASR_600', 'chi_diff', 'oxygen_vacancy',
                           'x_A_substitution', 'y_B_substitution', 'alpha_ratio_HT_LT']
        
        available_cluster = [f for f in cluster_features if f in df_features.columns]
        
        selected_cluster = st.multiselect(
            "Select features for clustering",
            available_cluster,
            default=available_cluster[:min(4, len(available_cluster))]
        )
        
        n_clusters = st.slider("Number of clusters", 2, 6, 4)
        
        if len(selected_cluster) >= 2 and len(df_features) >= n_clusters:
            fig, clusters, kmeans = perform_clustering_analysis(df_features, selected_cluster, n_clusters)
            
            if fig:
                st.pyplot(fig)
                plt.close()
                
                # Show cluster composition
                st.markdown("### 📊 Cluster Composition")
                
                cluster_summary = []
                for cluster_id in range(n_clusters):
                    cluster_mask = clusters == cluster_id
                    cluster_data = df_features[cluster_mask]
                    
                    summary = {
                        'Cluster': cluster_id,
                        'Size': len(cluster_data),
                        'Avg σ(600°C)': cluster_data['sigma_600'].mean() if 'sigma_600' in cluster_data else np.nan,
                        'Avg ASR(600°C)': cluster_data['ASR_600'].mean() if 'ASR_600' in cluster_data else np.nan,
                        'Avg χ_diff': cluster_data['chi_diff'].mean() if 'chi_diff' in cluster_data else np.nan,
                        'Avg Oxygen Vacancy': cluster_data['oxygen_vacancy'].mean() if 'oxygen_vacancy' in cluster_data else np.nan
                    }
                    cluster_summary.append(summary)
                
                st.dataframe(pd.DataFrame(cluster_summary), use_container_width=True)
                
                # Show representative materials per cluster
                st.markdown("### 🧪 Representative Materials per Cluster")
                
                for cluster_id in range(n_clusters):
                    cluster_mask = clusters == cluster_id
                    cluster_data = df_features[cluster_mask]
                    
                    with st.expander(f"Cluster {cluster_id} (n={len(cluster_data)})"):
                        display_cols = ['A_prime', 'B_prime', 'y_B_substitution', 
                                       'sigma_600', 'ASR_600', 'chi_diff']
                        available_display = [c for c in display_cols if c in cluster_data.columns]
                        st.dataframe(cluster_data[available_display].head(10), use_container_width=True)
    
    # =========================================================================
    # Page 7: Correlation Analysis
    # =========================================================================
    elif page == "📊 Correlation Analysis":
        st.markdown("## 📊 Property Correlation Analysis")
        
        st.markdown("""
        <div class="card">
            <p>Correlation matrix reveals relationships between composition, structure, and properties.
            Strong correlations (|r| > 0.7) suggest predictive relationships.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Select features for correlation
        all_corr_features = ['sigma_600', 'sigma_max', 'ASR_600', 'ASR_700',
                            'chi_diff', 'oxygen_vacancy', 'x_A_substitution', 
                            'y_B_substitution', 'alpha_ratio_HT_LT', 
                            'polarizability_avg', 'mass_ratio_AB']
        
        available_corr = [f for f in all_corr_features if f in df_features.columns]
        
        selected_corr = st.multiselect(
            "Select properties for correlation analysis",
            available_corr,
            default=available_corr[:min(8, len(available_corr))]
        )
        
        if len(selected_corr) >= 2:
            fig = create_correlation_heatmap(df_features, selected_corr)
            if fig:
                st.pyplot(fig)
                plt.close()
            
            # Highlight correlations with target properties
            st.markdown("### 🔍 Key Correlations with Conductivity")
            
            if 'sigma_600' in selected_corr:
                corr_with_sigma = df_features[selected_corr].corr()['sigma_600'].sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = ['green' if c > 0 else 'red' for c in corr_with_sigma.values]
                bars = ax.barh(range(len(corr_with_sigma)), corr_with_sigma.values, color=colors, alpha=0.7)
                
                ax.set_yticks(range(len(corr_with_sigma)))
                ax.set_yticklabels(corr_with_sigma.index)
                ax.set_xlabel('Correlation with σ(600°C)', fontsize=12)
                ax.set_title('Factors Affecting Electrical Conductivity', fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linewidth=0.5)
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Moderate')
                ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Strong')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='x')
                
                st.pyplot(fig)
                plt.close()
            
            if 'ASR_600' in selected_corr:
                st.markdown("### 🔍 Key Correlations with ASR")
                
                corr_with_asr = df_features[selected_corr].corr()['ASR_600'].sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = ['green' if c > 0 else 'red' for c in corr_with_asr.values]
                bars = ax.barh(range(len(corr_with_asr)), corr_with_asr.values, color=colors, alpha=0.7)
                
                ax.set_yticks(range(len(corr_with_asr)))
                ax.set_yticklabels(corr_with_asr.index)
                ax.set_xlabel('Correlation with ASR(600°C)', fontsize=12)
                ax.set_title('Factors Affecting Area-Specific Resistance', fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linewidth=0.5)
                ax.grid(True, alpha=0.3, axis='x')
                
                st.pyplot(fig)
                plt.close()
    
    # =========================================================================
    # Page 8: About
    # =========================================================================
    else:
        st.markdown("## ℹ️ About BaFeO₃ Multi-Property Predictor")
        
        st.markdown("""
        <div class="card">
            <h3>Advanced Tool for Doped Ferrite Analysis</h3>
            <p>Version 1.2 includes critical fixes for physical constraints and robust data handling.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>🎯 Key Features</h4>
                <ul>
                    <li><b>Electronegativity-based descriptors</b> (no ionic radii)</li>
                    <li><b>Multi-target prediction</b>: σ, ASR, Power, α</li>
                    <li><b>Ensemble ML models</b>: XGBoost + Random Forest</li>
                    <li><b>Property maps</b> for composition optimization</li>
                    <li><b>Bubble charts with heatmap overlays</b></li>
                    <li><b>Clustering analysis</b> for material families</li>
                    <li><b>Parallel coordinates</b> for multi-dimensional analysis</li>
                    <li><b>Fast mode</b> for quick training (3-5x speedup)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>🔬 Physical Descriptors</h4>
                <ul>
                    <li><b>χ(A), χ(B)</b>: Average A/B-site electronegativity</li>
                    <li><b>|χ(A)-χ(B)|</b>: Charge transfer potential</li>
                    <li><b>Oxygen vacancy</b>: [V_O] ≈ (y+α)/2</li>
                    <li><b>Polarizability</b>: Ionic polarizability average</li>
                    <li><b>Mass ratio</b>: B/A site mass ratio</li>
                    <li><b>α_HT/α_LT</b>: Chemical expansion indicator (constrained to 0.5-3.0)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>📊 Target Properties</h4>
            <ul>
                <li><b>σ(600°C)</b>: Electrical conductivity at 600°C (S/cm) - higher is better</li>
                <li><b>σmax</b>: Maximum conductivity in 500-700°C range (S/cm)</li>
                <li><b>ASR(600°C)</b>: Area-specific resistance (Ω·cm²) - lower is better, physically ≥0</li>
                <li><b>P(FC), 600°C</b>: Fuel cell power density (mW/cm²)</li>
                <li><b>α_HT/α_LT</b>: Thermal expansion ratio - indicates chemical expansion, physically 0.5-3.0</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>⚡ What's New in v1.2</h4>
            <ul>
                <li><b>Fixed α_HT/α_LT:</b> Physical constraint (0.5-3.0) applied - no more unphysical values</li>
                <li><b>Fixed ASR:</b> Negative values are now properly handled and removed</li>
                <li><b>Fixed interpolation errors:</b> Synchronized array lengths in create_property_map</li>
                <li><b>Fixed NaN handling:</b> Robust detection and removal of invalid data points</li>
                <li><b>Fixed hexbin dimension mismatch:</b> Proper array length synchronization</li>
                <li><b>Added physical constraints to predictions:</b> ASR ≥ 0, α_HT/α_LT ∈ [0.5, 3.0], σ ≥ 0</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="footer">
            <p>© 2025 BaFeO₃ Multi-Property Predictor | Developed for Materials Science Research</p>
            <p>Uses Pauling electronegativities | Ensemble machine learning | Scientific publication quality graphics</p>
            <p>Version 1.2 | Fixed physical constraints and interpolation errors</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Run the app
# =============================================================================
if __name__ == "__main__":
    main()

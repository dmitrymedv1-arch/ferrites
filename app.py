"""
BaFeO3 Electrode Materials Analyzer - Version 1.0
Advanced tool for analyzing and predicting properties of doped BaFeO3-based 
electrode materials for SOFC applications with multi-objective optimization.
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
from sklearn.preprocessing import MinMaxScaler
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
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

# =============================================================================
# Modern scientific color palette and styling
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
    'grid': '#dee2e6',
    'A_only': '#2ecc71',
    'B_only': '#e74c3c',
    'AB': '#3498db',
    'A_defect': '#f39c12',
    'B_defect': '#9b59b6'
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
# Enhanced electronegativity database (Pauling scale)
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
    'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Sb': 2.05,
    'Te': 2.10, 'Po': 2.00, 'At': 2.20, 'Rn': None, 'Fr': 0.70, 'Ra': 0.89,
    'Ac': 1.10, 'Th': 1.30, 'Pa': 1.50, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28,
    'Am': 1.13, 'Cm': 1.28, 'Bk': 1.30, 'Cf': 1.30, 'Es': 1.30, 'Fm': 1.30,
    'Md': 1.30, 'No': 1.30, 'Lr': 1.30
}

# =============================================================================
# Additional properties database (polarizability, ionization potential)
# =============================================================================
IONIC_POLARIZABILITY = {
    'Ba': 1.55, 'Sr': 0.86, 'Ca': 0.47, 'La': 1.04, 'Y': 0.55,
    'Ti': 0.19, 'Zr': 0.37, 'Sn': 0.24, 'Ce': 0.43, 'Hf': 0.34,
    'Sc': 0.29, 'In': 0.42, 'Yb': 0.31, 'Gd': 0.39, 'Al': 0.05,
    'Ga': 0.16, 'Fe': 0.18, 'Co': 0.15, 'Ni': 0.13, 'Zn': 0.28,
    'Mg': 0.09, 'O': 2.0, 'Pr': 0.45, 'Nd': 0.48, 'Sm': 0.52,
    'Eu': 0.55, 'Dy': 0.58, 'Ho': 0.60, 'Er': 0.62, 'Tm': 0.64,
    'Lu': 0.66, 'Bi': 0.85, 'Mo': 0.35, 'W': 0.40, 'Cu': 0.22,
    'Ni': 0.13, 'Co': 0.15, 'Mn': 0.20, 'Cr': 0.18, 'V': 0.22,
    'Nb': 0.38, 'Ta': 0.42, 'Ru': 0.28, 'Rh': 0.30, 'Pd': 0.32,
    'Ag': 0.34, 'Cd': 0.36, 'In': 0.42, 'Sn': 0.24, 'Sb': 0.26
}

IONIZATION_POTENTIAL = {
    'Ba': 5.21, 'Sr': 5.69, 'Ca': 6.11, 'La': 5.58, 'Y': 6.22,
    'Ti': 6.82, 'Zr': 6.84, 'Sn': 7.34, 'Ce': 5.54, 'Hf': 6.83,
    'Sc': 6.56, 'In': 5.79, 'Yb': 6.25, 'Gd': 6.15, 'Al': 5.99,
    'Ga': 6.00, 'Fe': 7.90, 'Co': 7.88, 'Ni': 7.64, 'Zn': 9.39,
    'Mg': 7.65, 'O': 13.62, 'Pr': 5.46, 'Nd': 5.52, 'Sm': 5.64,
    'Eu': 5.67, 'Dy': 5.93, 'Ho': 6.02, 'Er': 6.11, 'Tm': 6.18,
    'Lu': 6.18, 'Bi': 7.29, 'Mo': 7.09, 'W': 7.98, 'Cu': 7.73,
    'Ni': 7.64, 'Co': 7.88, 'Mn': 7.43, 'Cr': 6.77, 'V': 6.75,
    'Nb': 6.88, 'Ta': 7.55, 'Ru': 7.36, 'Rh': 7.46, 'Pd': 8.34,
    'Ag': 7.58, 'Cd': 8.99, 'In': 5.79, 'Sn': 7.34, 'Sb': 8.64
}

# =============================================================================
# Helper functions for property extraction
# =============================================================================
def get_electronegativity(element):
    """Get Pauling electronegativity for element"""
    if element == '-' or element is None or pd.isna(element):
        return None
    return ELECTRONEGATIVITY.get(str(element).strip(), None)

def get_polarizability(element):
    """Get ionic polarizability"""
    if element == '-' or element is None or pd.isna(element):
        return 0
    return IONIC_POLARIZABILITY.get(str(element).strip(), 0)

def get_ionization_potential(element):
    """Get ionization potential"""
    if element == '-' or element is None or pd.isna(element):
        return 0
    return IONIZATION_POTENTIAL.get(str(element).strip(), 0)

def determine_doping_type(row):
    """
    Determine doping strategy based on A and B site composition
    Returns: 'A_only', 'B_only', 'AB', 'A_defect', 'B_defect'
    """
    x = row.get('x', 0)
    y = row.get('y', 0)
    z = row.get('z', 1)
    alpha = row.get('α', 0)
    
    has_A_dopant = x > 0 if not pd.isna(x) else False
    has_B_dopant = (y > 0 if not pd.isna(y) else False) or (alpha > 0 if not pd.isna(alpha) else False)
    has_A_deficit = z < 1 if not pd.isna(z) else False
    has_B_deficit = (z < 1 and 'Fe' in str(row.get('B\'', ''))) or False
    
    if has_A_deficit and not has_B_deficit:
        return 'A_defect'
    elif has_B_deficit and not has_A_deficit:
        return 'B_defect'
    elif has_A_dopant and has_B_dopant:
        return 'AB'
    elif has_A_dopant and not has_B_dopant:
        return 'A_only'
    elif has_B_dopant and not has_A_dopant:
        return 'B_only'
    else:
        return 'undoped'

def calculate_descriptors_BaFeO3(row):
    """
    Calculate enhanced descriptors for BaFeO3-based materials
    WITHOUT ionic radii and tolerance factor
    """
    descriptors = {}
    
    # Extract composition parameters
    A_elements = []
    B_elements = []
    
    # A-site elements
    for col in ['A\'', 'A\'\'']:
        val = row.get(col)
        if val and val != '-' and not pd.isna(val):
            A_elements.append(str(val))
    
    # B-site elements
    for col in ['B\'', 'B\'\'', 'B\'\'\'', 'B\'\'\'"']:
        val = row.get(col)
        if val and val != '-' and not pd.isna(val):
            B_elements.append(str(val))
    
    # Get concentrations
    x = row.get('x', 0)
    y = row.get('y', 0)
    z = row.get('z', 1)
    alpha = row.get('α', 0)
    
    descriptors['x_A'] = x if not pd.isna(x) else 0
    descriptors['y_B'] = y if not pd.isna(y) else 0
    descriptors['alpha_B'] = alpha if not pd.isna(alpha) else 0
    descriptors['z_stoichiometry'] = z if not pd.isna(z) else 1
    
    # Total doping concentrations
    descriptors['total_A_doping'] = x if not pd.isna(x) else 0
    descriptors['total_B_doping'] = (y if not pd.isna(y) else 0) + (alpha if not pd.isna(alpha) else 0)
    
    # Deficit indicators
    descriptors['deficit_A'] = 1 if (z < 1 and not pd.isna(z)) else 0
    descriptors['deficit_B'] = 1 if (z < 1 and 'Fe' in str(row.get('B\'', ''))) else 0
    descriptors['deficit_magnitude'] = 1 - z if (z < 1 and not pd.isna(z)) else 0
    
    # Number of different elements
    descriptors['n_A_elements'] = len([e for e in A_elements if e and e != '-'])
    descriptors['n_B_elements'] = len([e for e in B_elements if e and e != '-'])
    
    # Doping type classification
    descriptors['doping_type'] = determine_doping_type(row)
    
    # Electronegativity descriptors
    en_A_list = []
    for elem in A_elements:
        en = get_electronegativity(elem)
        if en is not None:
            en_A_list.append(en)
    
    en_B_list = []
    for elem in B_elements:
        en = get_electronegativity(elem)
        if en is not None:
            en_B_list.append(en)
    
    descriptors['avg_EN_A'] = np.mean(en_A_list) if en_A_list else get_electronegativity('Ba')
    descriptors['avg_EN_B'] = np.mean(en_B_list) if en_B_list else get_electronegativity('Fe')
    descriptors['min_EN_A'] = min(en_A_list) if en_A_list else descriptors['avg_EN_A']
    descriptors['max_EN_A'] = max(en_A_list) if en_A_list else descriptors['avg_EN_A']
    descriptors['min_EN_B'] = min(en_B_list) if en_B_list else descriptors['avg_EN_B']
    descriptors['max_EN_B'] = max(en_B_list) if en_B_list else descriptors['avg_EN_B']
    descriptors['EN_diff_A'] = descriptors['max_EN_A'] - descriptors['min_EN_A']
    descriptors['EN_diff_B'] = descriptors['max_EN_B'] - descriptors['min_EN_B']
    descriptors['EN_diff_AB'] = descriptors['avg_EN_B'] - descriptors['avg_EN_A']
    descriptors['EN_diff_BO'] = descriptors['avg_EN_B'] - get_electronegativity('O')
    
    # Polarizability descriptors
    pol_A_list = [get_polarizability(e) for e in A_elements]
    pol_B_list = [get_polarizability(e) for e in B_elements]
    
    descriptors['avg_polarizability_A'] = np.mean(pol_A_list) if pol_A_list else 0
    descriptors['avg_polarizability_B'] = np.mean(pol_B_list) if pol_B_list else 0
    descriptors['total_polarizability'] = descriptors['avg_polarizability_A'] + descriptors['avg_polarizability_B']
    
    # Ionization potential descriptors
    ip_A_list = [get_ionization_potential(e) for e in A_elements]
    ip_B_list = [get_ionization_potential(e) for e in B_elements]
    
    descriptors['avg_IP_A'] = np.mean(ip_A_list) if ip_A_list else 0
    descriptors['avg_IP_B'] = np.mean(ip_B_list) if ip_B_list else 0
    descriptors['IP_diff_AB'] = descriptors['avg_IP_B'] - descriptors['avg_IP_A']
    
    # Lattice parameters (if available)
    descriptors['a'] = row.get('a') if not pd.isna(row.get('a')) else None
    descriptors['b'] = row.get('b') if not pd.isna(row.get('b')) else None
    descriptors['c'] = row.get('c') if not pd.isna(row.get('c')) else None
    descriptors['V'] = row.get('V') if not pd.isna(row.get('V')) else None
    
    # Conductivity values
    descriptors['cond_500_ox'] = row.get('500 °C_ox') if not pd.isna(row.get('500 °C_ox')) else None
    descriptors['cond_600_ox'] = row.get('600 °C_ox') if not pd.isna(row.get('600 °C_ox')) else None
    descriptors['cond_700_ox'] = row.get('700 °C_ox') if not pd.isna(row.get('700 °C_ox')) else None
    
    descriptors['cond_500_red'] = row.get('500 °C_red') if not pd.isna(row.get('500 °C_red')) else None
    descriptors['cond_600_red'] = row.get('600 °C_red') if not pd.isna(row.get('600 °C_red')) else None
    descriptors['cond_700_red'] = row.get('700 °C_red') if not pd.isna(row.get('700 °C_red')) else None
    
    # TEC values
    descriptors['TEC_LT'] = row.get('αLT') if not pd.isna(row.get('αLT')) else None
    descriptors['TEC_HT'] = row.get('αHT') if not pd.isna(row.get('αHT')) else None
    descriptors['TEC_avg'] = row.get('αav') if not pd.isna(row.get('αav')) else None
    descriptors['TEC_delta'] = (descriptors['TEC_HT'] - descriptors['TEC_LT']) if (descriptors['TEC_HT'] and descriptors['TEC_LT']) else None
    
    # Power density
    descriptors['power_600'] = row.get('600_power') if not pd.isna(row.get('600_power')) else None
    descriptors['power_650'] = row.get('650_power') if not pd.isna(row.get('650_power')) else None
    descriptors['power_700'] = row.get('700_power') if not pd.isna(row.get('700_power')) else None
    
    # ASR values
    descriptors['ASR_600'] = row.get('600_ASR') if not pd.isna(row.get('600_ASR')) else None
    descriptors['ASR_650'] = row.get('650_ASR') if not pd.isna(row.get('650_ASR')) else None
    descriptors['ASR_700'] = row.get('700_ASR') if not pd.isna(row.get('700_ASR')) else None
    
    return descriptors

# =============================================================================
# Excel loading function with proper header handling
# =============================================================================
@st.cache_data
def load_excel_data(uploaded_file):
    """
    Load Excel file with proper structure:
    Row 0: Parameter names (Chemical composition, lattice/ox, etc.)
    Row 1: Detailed column descriptions
    Row 2+: Actual data
    """
    if uploaded_file is None:
        return None
    
    try:
        # Read the Excel file without assuming headers
        df_raw = pd.read_excel(uploaded_file, sheet_name='электроды', header=None)
        
        # The actual data starts from row 2 (index 2) after two header rows
        if len(df_raw) < 3:
            st.error("File has fewer than 3 rows. Expected header rows + data.")
            return None
        
        # Get column names from row 0 and row 1
        header_row0 = df_raw.iloc[0].fillna('').astype(str).tolist()
        header_row1 = df_raw.iloc[1].fillna('').astype(str).tolist()
        
        # Combine headers: use row1 if meaningful, otherwise row0
        combined_headers = []
        for i, (h0, h1) in enumerate(zip(header_row0, header_row1)):
            if h1 and h1 != 'nan' and not h1.startswith('Unnamed'):
                combined_headers.append(f"{h1}")
            elif h0 and h0 != 'nan':
                combined_headers.append(h0)
            else:
                combined_headers.append(f"col_{i}")
        
        # Clean up headers: remove duplicates, handle special chars
        cleaned_headers = []
        seen = set()
        for h in combined_headers:
            if h in seen:
                counter = 1
                new_h = f"{h}_{counter}"
                while new_h in seen:
                    counter += 1
                    new_h = f"{h}_{counter}"
                cleaned_headers.append(new_h)
                seen.add(new_h)
            else:
                cleaned_headers.append(h)
                seen.add(h)
        
        # Create DataFrame with cleaned headers
        df = pd.DataFrame(df_raw.iloc[2:].values, columns=cleaned_headers)
        
        # Rename columns to standard names for easier access
        column_mapping = {
            'Composition': 'Composition',
            'A\'': 'A\'',
            'A\'\'': 'A\'\'',
            'B\'': 'B\'',
            'B\'\'': 'B\'\'',
            'B\'\'\'': 'B\'\'\'',
            'B\'\'\'"': 'B\'\'\'"',
            'x': 'x',
            'y': 'y',
            'z': 'z',
            'α': 'α',
            'a': 'a',
            'b': 'b',
            'c': 'c',
            'V': 'V',
            'Vpseud': 'Vpseud',
            'apseud': 'apseud',
            '500 °C': '500 °C_ox',
            '600 °C': '600 °C_ox',
            '700 °C': '700 °C_ox',
            'Tmax': 'Tmax',
            '500 °C_red': '500 °C_red',
            '600 °C_red': '600 °C_red',
            '700 °C_red': '700 °C_red',
            'αLT': 'αLT',
            'αHT': 'αHT',
            'αav': 'αav',
            '600_power': '600_power',
            '650_power': '650_power',
            '700_power': '700_power',
            '600_ASR': '600_ASR',
            '650_ASR': '650_ASR',
            '700_ASR': '700_ASR',
            'doi': 'doi'
        }
        
        # Apply mapping for columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['x', 'y', 'z', 'α', 'a', 'b', 'c', 'V', 'Vpseud', 'apseud',
                       '500 °C_ox', '600 °C_ox', '700 °C_ox', 'Tmax',
                       '500 °C_red', '600 °C_red', '700 °C_red',
                       'αLT', 'αHT', 'αav', '600_power', '650_power', '700_power',
                       '600_ASR', '650_ASR', '700_ASR']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add derived columns
        df['doping_type'] = df.apply(determine_doping_type, axis=1)
        
        # Calculate descriptors for each row
        descriptors_list = []
        for idx, row in df.iterrows():
            desc = calculate_descriptors_BaFeO3(row)
            descriptors_list.append(desc)
        
        df_descriptors = pd.DataFrame(descriptors_list)
        
        # Combine original data with descriptors
        for col in df_descriptors.columns:
            if col not in df.columns:
                df[col] = df_descriptors[col].values
        
        return df
    
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

# =============================================================================
# Multi-objective optimization functions
# =============================================================================
def calculate_composite_score(row, weights):
    """
    Calculate weighted composite score for multi-objective optimization
    weights: dict with keys 'conductivity', 'power', 'ASR', 'TEC', 'TEC_stability'
    """
    score = 0
    total_weight = 0
    
    # Conductivity (higher is better) - use 700°C as reference
    if weights.get('conductivity', 0) > 0:
        cond_val = row.get('cond_700_ox')
        if cond_val and not pd.isna(cond_val) and cond_val > 0:
            # Normalize conductivity (typical range 0-200 S/cm)
            norm_cond = min(cond_val / 200, 1.0)
            score += weights['conductivity'] * norm_cond
        total_weight += weights['conductivity']
    
    # Power density (higher is better)
    if weights.get('power', 0) > 0:
        power_val = row.get('power_700')
        if power_val and not pd.isna(power_val) and power_val > 0:
            # Normalize power (typical range 0-1500 mW/cm²)
            norm_power = min(power_val / 1500, 1.0)
            score += weights['power'] * norm_power
        total_weight += weights['power']
    
    # ASR (lower is better) - invert
    if weights.get('ASR', 0) > 0:
        asr_val = row.get('ASR_700')
        if asr_val and not pd.isna(asr_val) and asr_val > 0:
            # Normalize ASR (typical range 0-10 Ohm·cm²)
            norm_asr = max(1 - min(asr_val / 10, 1.0), 0)
            score += weights['ASR'] * norm_asr
        total_weight += weights['ASR']
    
    # Average TEC (lower is better) - invert
    if weights.get('TEC', 0) > 0:
        tec_val = row.get('TEC_avg')
        if tec_val and not pd.isna(tec_val) and tec_val > 0:
            # Normalize TEC (typical range 10-30 ×10⁻⁶ K⁻¹)
            norm_tec = max(1 - min((tec_val - 10) / 20, 1.0), 0)
            score += weights['TEC'] * norm_tec
        total_weight += weights['TEC']
    
    # TEC stability (low delta between LT and HT is better)
    if weights.get('TEC_stability', 0) > 0:
        tec_lt = row.get('TEC_LT')
        tec_ht = row.get('TEC_HT')
        if tec_lt and tec_ht and not pd.isna(tec_lt) and not pd.isna(tec_ht):
            tec_delta = abs(tec_ht - tec_lt)
            # Normalize delta (typical range 0-15)
            norm_delta = max(1 - min(tec_delta / 15, 1.0), 0)
            score += weights['TEC_stability'] * norm_delta
        total_weight += weights['TEC_stability']
    
    if total_weight > 0:
        return score / total_weight
    return 0

def get_optimal_compositions(df, weights, top_n=10):
    """Return top N compositions based on weighted score"""
    df_copy = df.copy()
    df_copy['composite_score'] = df_copy.apply(lambda row: calculate_composite_score(row, weights), axis=1)
    df_sorted = df_copy.sort_values('composite_score', ascending=False)
    return df_sorted.head(top_n)

# =============================================================================
# Visualization functions for doping strategy analysis
# =============================================================================
def create_doping_strategy_comparison(df):
    """Create comprehensive comparison of different doping strategies"""
    
    fig = plt.figure(figsize=(20, 14))
    
    # Get unique doping types
    doping_types = df['doping_type'].unique()
    colors = [MODERN_COLORS.get(dt, MODERN_COLORS['gray']) for dt in doping_types]
    
    # 1. Conductivity comparison (700°C)
    ax1 = fig.add_subplot(2, 3, 1)
    data_to_plot = []
    for dt in doping_types:
        data = df[df['doping_type'] == dt]['cond_700_ox'].dropna()
        if len(data) > 0:
            data_to_plot.append(data.values)
    
    if data_to_plot:
        bp = ax1.boxplot(data_to_plot, labels=doping_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('Conductivity at 700°C (S/cm)', fontsize=11)
        ax1.set_title('Conductivity by Doping Strategy', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Power density comparison (700°C)
    ax2 = fig.add_subplot(2, 3, 2)
    data_to_plot = []
    for dt in doping_types:
        data = df[df['doping_type'] == dt]['power_700'].dropna()
        if len(data) > 0:
            data_to_plot.append(data.values)
    
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=doping_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Power Density at 700°C (mW/cm²)', fontsize=11)
        ax2.set_title('Power Density by Doping Strategy', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. ASR comparison (700°C)
    ax3 = fig.add_subplot(2, 3, 3)
    data_to_plot = []
    for dt in doping_types:
        data = df[df['doping_type'] == dt]['ASR_700'].dropna()
        if len(data) > 0:
            data_to_plot.append(data.values)
    
    if data_to_plot:
        bp = ax3.boxplot(data_to_plot, labels=doping_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_ylabel('ASR at 700°C (Ω·cm²)', fontsize=11)
        ax3.set_title('ASR by Doping Strategy', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. TEC average comparison
    ax4 = fig.add_subplot(2, 3, 4)
    data_to_plot = []
    for dt in doping_types:
        data = df[df['doping_type'] == dt]['TEC_avg'].dropna()
        if len(data) > 0:
            data_to_plot.append(data.values)
    
    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=doping_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax4.set_ylabel('TEC (×10⁻⁶ K⁻¹)', fontsize=11)
        ax4.set_title('Average TEC by Doping Strategy', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. TEC stability (ΔTEC = HT - LT)
    ax5 = fig.add_subplot(2, 3, 5)
    data_to_plot = []
    for dt in doping_types:
        subset = df[df['doping_type'] == dt]
        tec_delta = (subset['TEC_HT'] - subset['TEC_LT']).dropna()
        if len(tec_delta) > 0:
            data_to_plot.append(tec_delta.values)
    
    if data_to_plot:
        bp = ax5.boxplot(data_to_plot, labels=doping_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.set_ylabel('ΔTEC (×10⁻⁶ K⁻¹)', fontsize=11)
        ax5.set_title('TEC Stability (HT - LT)', fontsize=12, fontweight='bold')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
    
    # 6. Sample count
    ax6 = fig.add_subplot(2, 3, 6)
    counts = []
    for dt in doping_types:
        counts.append(len(df[df['doping_type'] == dt]))
    bars = ax6.bar(doping_types, counts, color=colors[:len(doping_types)], edgecolor='black', alpha=0.7)
    ax6.set_ylabel('Number of Samples', fontsize=11)
    ax6.set_title('Dataset Size by Doping Strategy', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), 
                ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Doping Strategy Comparison: A-site vs B-site vs Co-doping vs Defects', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_radar_chart_strategies(df):
    """Create radar chart comparing average properties of each doping strategy"""
    
    doping_types = df['doping_type'].unique()
    
    # Define properties to compare (normalized)
    properties = {
        'Conductivity\n700°C': 'cond_700_ox',
        'Power\n700°C': 'power_700',
        'ASR\n700°C (inv)': 'ASR_700',
        'TEC\navg (inv)': 'TEC_avg',
        'TEC\nstability': 'TEC_stability'
    }
    
    # Calculate normalized averages for each strategy
    strategy_data = {}
    
    for dt in doping_types:
        subset = df[df['doping_type'] == dt]
        values = []
        
        # Conductivity (higher better)
        cond_vals = subset['cond_700_ox'].dropna()
        if len(cond_vals) > 0:
            cond_mean = cond_vals.mean()
            norm_cond = min(cond_mean / 200, 1.0)
        else:
            norm_cond = 0
        values.append(norm_cond)
        
        # Power (higher better)
        power_vals = subset['power_700'].dropna()
        if len(power_vals) > 0:
            power_mean = power_vals.mean()
            norm_power = min(power_mean / 1500, 1.0)
        else:
            norm_power = 0
        values.append(norm_power)
        
        # ASR (lower better - invert)
        asr_vals = subset['ASR_700'].dropna()
        if len(asr_vals) > 0:
            asr_mean = asr_vals.mean()
            norm_asr = max(1 - min(asr_mean / 10, 1.0), 0)
        else:
            norm_asr = 0
        values.append(norm_asr)
        
        # TEC average (lower better - invert)
        tec_vals = subset['TEC_avg'].dropna()
        if len(tec_vals) > 0:
            tec_mean = tec_vals.mean()
            norm_tec = max(1 - min((tec_mean - 10) / 20, 1.0), 0)
        else:
            norm_tec = 0
        values.append(norm_tec)
        
        # TEC stability (lower delta better - invert)
        tec_delta_vals = (subset['TEC_HT'] - subset['TEC_LT']).dropna()
        if len(tec_delta_vals) > 0:
            delta_mean = abs(tec_delta_vals.mean())
            norm_delta = max(1 - min(delta_mean / 15, 1.0), 0)
        else:
            norm_delta = 0
        values.append(norm_delta)
        
        strategy_data[dt] = values
    
    # Create radar chart
    categories = list(properties.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = [MODERN_COLORS.get(dt, MODERN_COLORS['gray']) for dt in strategy_data.keys()]
    
    for (dt, values), color in zip(strategy_data.items(), colors):
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=2, label=dt, color=color)
        ax.fill(angles, values_plot, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.set_title('Doping Strategy Performance Radar Chart\n(Normalized, higher = better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True)
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for all key properties and descriptors"""
    
    # Select relevant columns for correlation
    correlation_cols = [
        'x_A', 'total_B_doping', 'deficit_magnitude',
        'avg_EN_A', 'avg_EN_B', 'EN_diff_AB', 'EN_diff_BO',
        'avg_polarizability_A', 'avg_polarizability_B',
        'cond_500_ox', 'cond_600_ox', 'cond_700_ox',
        'power_600', 'power_650', 'power_700',
        'ASR_600', 'ASR_650', 'ASR_700',
        'TEC_LT', 'TEC_HT', 'TEC_avg', 'TEC_delta'
    ]
    
    # Filter available columns
    available_cols = [col for col in correlation_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True, ax=ax,
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                annot_kws={'size': 8})
    
    ax.set_title('Property Correlation Matrix', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, ha='right', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    return fig

def create_parallel_coordinates_optimization(df, weights):
    """Create parallel coordinates plot for multi-objective optimization"""
    
    # Calculate composite score
    df_copy = df.copy()
    df_copy['composite_score'] = df_copy.apply(lambda row: calculate_composite_score(row, weights), axis=1)
    
    # Select top 30 materials for visualization
    df_top = df_copy.nlargest(30, 'composite_score')
    
    # Select features for parallel coordinates
    features = ['total_B_doping', 'avg_EN_B', 'cond_700_ox', 'power_700', 'ASR_700', 'TEC_avg', 'composite_score']
    available_features = [f for f in features if f in df_top.columns]
    
    if len(available_features) < 2:
        return None
    
    # Normalize data for visualization
    df_norm = df_top[available_features].copy()
    for col in available_features[:-1]:
        if df_norm[col].max() > df_norm[col].min():
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create parallel coordinates
    x_coords = np.arange(len(available_features))
    
    # Color by composite score
    norm = plt.Normalize(df_norm['composite_score'].min(), df_norm['composite_score'].max())
    colors = plt.cm.viridis(norm(df_norm['composite_score']))
    
    for idx in range(len(df_norm)):
        y_coords = df_norm.iloc[idx, :].values
        ax.plot(x_coords, y_coords, color=colors[idx], alpha=0.5, linewidth=1)
    
    # Add mean lines for top performers
    top_quartile = df_norm[df_norm['composite_score'] > df_norm['composite_score'].quantile(0.75)]
    if len(top_quartile) > 0:
        mean_values = top_quartile[available_features].mean()
        ax.plot(x_coords, mean_values, 'r-', linewidth=3, label='Top 25% performers', alpha=0.8)
    
    ax.set_xticks(x_coords)
    ax.set_xticklabels([f.replace('_', ' ').title() for f in available_features], rotation=45, ha='right')
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_title('Parallel Coordinates: Top Performing Materials\n(Higher composite score = better overall)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Composite Score', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_dopant_concentration_effects(df):
    """Create plots showing effects of dopant concentration on properties"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. B-site doping vs Conductivity at 700°C
    ax1 = axes[0, 0]
    b_only = df[df['doping_type'] == 'B_only']
    if len(b_only) > 0:
        ax1.scatter(b_only['total_B_doping'], b_only['cond_700_ox'], 
                   alpha=0.7, s=80, c=MODERN_COLORS['B_only'], 
                   edgecolors='black', linewidth=0.5, label='B-only doping')
        
        # Add trend line
        if len(b_only) > 2:
            valid = b_only[['total_B_doping', 'cond_700_ox']].dropna()
            if len(valid) > 2:
                z = np.polyfit(valid['total_B_doping'], valid['cond_700_ox'], 1)
                x_line = np.linspace(valid['total_B_doping'].min(), valid['total_B_doping'].max(), 50)
                ax1.plot(x_line, np.polyval(z, x_line), '--', color=MODERN_COLORS['B_only'], alpha=0.5)
    
    ax1.set_xlabel('Total B-site Doping Concentration', fontsize=11)
    ax1.set_ylabel('Conductivity at 700°C (S/cm)', fontsize=11)
    ax1.set_title('Effect of B-site Doping on Conductivity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. A-site doping vs Power Density
    ax2 = axes[0, 1]
    a_only = df[df['doping_type'] == 'A_only']
    if len(a_only) > 0:
        ax2.scatter(a_only['x_A'], a_only['power_700'], 
                   alpha=0.7, s=80, c=MODERN_COLORS['A_only'], 
                   edgecolors='black', linewidth=0.5, label='A-only doping')
        
        if len(a_only) > 2:
            valid = a_only[['x_A', 'power_700']].dropna()
            if len(valid) > 2:
                z = np.polyfit(valid['x_A'], valid['power_700'], 1)
                x_line = np.linspace(valid['x_A'].min(), valid['x_A'].max(), 50)
                ax2.plot(x_line, np.polyval(z, x_line), '--', color=MODERN_COLORS['A_only'], alpha=0.5)
    
    ax2.set_xlabel('A-site Doping Concentration (x)', fontsize=11)
    ax2.set_ylabel('Power Density at 700°C (mW/cm²)', fontsize=11)
    ax2.set_title('Effect of A-site Doping on Power Density', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Co-doping vs ASR
    ax3 = axes[0, 2]
    ab_doping = df[df['doping_type'] == 'AB']
    if len(ab_doping) > 0:
        ax3.scatter(ab_doping['total_B_doping'], ab_doping['ASR_700'], 
                   alpha=0.7, s=80, c=MODERN_COLORS['AB'], 
                   edgecolors='black', linewidth=0.5, label='Co-doping (A+B)')
        
        if len(ab_doping) > 2:
            valid = ab_doping[['total_B_doping', 'ASR_700']].dropna()
            if len(valid) > 2:
                z = np.polyfit(valid['total_B_doping'], valid['ASR_700'], 1)
                x_line = np.linspace(valid['total_B_doping'].min(), valid['total_B_doping'].max(), 50)
                ax3.plot(x_line, np.polyval(z, x_line), '--', color=MODERN_COLORS['AB'], alpha=0.5)
    
    ax3.set_xlabel('Total B-site Doping Concentration', fontsize=11)
    ax3.set_ylabel('ASR at 700°C (Ω·cm²)', fontsize=11)
    ax3.set_title('Effect of Co-doping on ASR', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Defect concentration vs TEC
    ax4 = axes[1, 0]
    defect = df[df['deficit_A'] == 1]
    if len(defect) > 0:
        ax4.scatter(defect['deficit_magnitude'], defect['TEC_avg'], 
                   alpha=0.7, s=80, c=MODERN_COLORS['A_defect'], 
                   edgecolors='black', linewidth=0.5, label='A-site deficit')
    
    b_defect = df[df['deficit_B'] == 1]
    if len(b_defect) > 0:
        ax4.scatter(b_defect['deficit_magnitude'], b_defect['TEC_avg'], 
                   alpha=0.7, s=80, c=MODERN_COLORS['B_defect'], 
                   edgecolors='black', linewidth=0.5, label='B-site deficit')
    
    ax4.set_xlabel('Deficit Magnitude (1 - z)', fontsize=11)
    ax4.set_ylabel('Average TEC (×10⁻⁶ K⁻¹)', fontsize=11)
    ax4.set_title('Effect of Stoichiometric Deficit on TEC', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Electronegativity vs Conductivity
    ax5 = axes[1, 1]
    all_data = df.dropna(subset=['avg_EN_B', 'cond_700_ox'])
    if len(all_data) > 0:
        scatter = ax5.scatter(all_data['avg_EN_B'], all_data['cond_700_ox'], 
                             c=all_data['doping_type'].map(lambda x: MODERN_COLORS.get(x, MODERN_COLORS['gray'])),
                             s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        if len(all_data) > 2:
            z = np.polyfit(all_data['avg_EN_B'], all_data['cond_700_ox'], 1)
            x_line = np.linspace(all_data['avg_EN_B'].min(), all_data['avg_EN_B'].max(), 50)
            ax5.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5, linewidth=1.5)
            
            # Add correlation coefficient
            corr = all_data['avg_EN_B'].corr(all_data['cond_700_ox'])
            ax5.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax5.transAxes, 
                    fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    ax5.set_xlabel('Average B-site Electronegativity', fontsize=11)
    ax5.set_ylabel('Conductivity at 700°C (S/cm)', fontsize=11)
    ax5.set_title('Electronegativity vs Conductivity', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. EN difference vs Power Density
    ax6 = axes[1, 2]
    all_data2 = df.dropna(subset=['EN_diff_BO', 'power_700'])
    if len(all_data2) > 0:
        scatter = ax6.scatter(all_data2['EN_diff_BO'], all_data2['power_700'], 
                             c=all_data2['doping_type'].map(lambda x: MODERN_COLORS.get(x, MODERN_COLORS['gray'])),
                             s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        if len(all_data2) > 2:
            z = np.polyfit(all_data2['EN_diff_BO'], all_data2['power_700'], 1)
            x_line = np.linspace(all_data2['EN_diff_BO'].min(), all_data2['EN_diff_BO'].max(), 50)
            ax6.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5, linewidth=1.5)
            
            corr = all_data2['EN_diff_BO'].corr(all_data2['power_700'])
            ax6.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax6.transAxes, 
                    fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    ax6.set_xlabel('B-O Electronegativity Difference', fontsize=11)
    ax6.set_ylabel('Power Density at 700°C (mW/cm²)', fontsize=11)
    ax6.set_title('B-O Bond Covalency vs Power Density', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Dopant Concentration Effects on Material Properties', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_temperature_dependence_plots(df):
    """Create plots showing temperature dependence of properties"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    temperatures = [500, 600, 700]
    
    # 1. Conductivity vs Temperature (different doping strategies)
    ax1 = axes[0, 0]
    
    for dt in df['doping_type'].unique():
        subset = df[df['doping_type'] == dt]
        cond_means = []
        cond_stds = []
        
        for T in temperatures:
            col = f'cond_{T}_ox'
            if col in subset.columns:
                vals = subset[col].dropna()
                if len(vals) > 0:
                    cond_means.append(vals.mean())
                    cond_stds.append(vals.std())
                else:
                    cond_means.append(np.nan)
                    cond_stds.append(np.nan)
            else:
                cond_means.append(np.nan)
                cond_stds.append(np.nan)
        
        if not np.all(np.isnan(cond_means)):
            ax1.errorbar(temperatures[:len(cond_means)], cond_means, yerr=cond_stds, 
                        marker='o', linewidth=2, markersize=8, capsize=5,
                        label=dt, color=MODERN_COLORS.get(dt, MODERN_COLORS['gray']))
    
    ax1.set_xlabel('Temperature (°C)', fontsize=11)
    ax1.set_ylabel('Conductivity (S/cm)', fontsize=11)
    ax1.set_title('Conductivity vs Temperature by Doping Strategy', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Power Density vs Temperature
    ax2 = axes[0, 1]
    
    power_temps = [600, 650, 700]
    
    for dt in df['doping_type'].unique():
        subset = df[df['doping_type'] == dt]
        power_means = []
        power_stds = []
        
        for T in power_temps:
            col = f'power_{T}'
            if col in subset.columns:
                vals = subset[col].dropna()
                if len(vals) > 0:
                    power_means.append(vals.mean())
                    power_stds.append(vals.std())
                else:
                    power_means.append(np.nan)
                    power_stds.append(np.nan)
            else:
                power_means.append(np.nan)
                power_stds.append(np.nan)
        
        if not np.all(np.isnan(power_means)):
            ax2.errorbar(power_temps[:len(power_means)], power_means, yerr=power_stds, 
                        marker='s', linewidth=2, markersize=8, capsize=5,
                        label=dt, color=MODERN_COLORS.get(dt, MODERN_COLORS['gray']))
    
    ax2.set_xlabel('Temperature (°C)', fontsize=11)
    ax2.set_ylabel('Power Density (mW/cm²)', fontsize=11)
    ax2.set_title('Power Density vs Temperature by Doping Strategy', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. ASR vs Temperature
    ax3 = axes[1, 0]
    
    for dt in df['doping_type'].unique():
        subset = df[df['doping_type'] == dt]
        asr_means = []
        asr_stds = []
        
        for T in power_temps:
            col = f'ASR_{T}'
            if col in subset.columns:
                vals = subset[col].dropna()
                if len(vals) > 0:
                    asr_means.append(vals.mean())
                    asr_stds.append(vals.std())
                else:
                    asr_means.append(np.nan)
                    asr_stds.append(np.nan)
            else:
                asr_means.append(np.nan)
                asr_stds.append(np.nan)
        
        if not np.all(np.isnan(asr_means)):
            ax3.errorbar(power_temps[:len(asr_means)], asr_means, yerr=asr_stds, 
                        marker='^', linewidth=2, markersize=8, capsize=5,
                        label=dt, color=MODERN_COLORS.get(dt, MODERN_COLORS['gray']))
    
    ax3.set_xlabel('Temperature (°C)', fontsize=11)
    ax3.set_ylabel('ASR (Ω·cm²)', fontsize=11)
    ax3.set_title('ASR vs Temperature by Doping Strategy', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Conductivity ratio (ox/red) at different temperatures
    ax4 = axes[1, 1]
    
    for dt in df['doping_type'].unique():
        subset = df[df['doping_type'] == dt]
        ratios = []
        
        for T in temperatures:
            cond_ox_col = f'cond_{T}_ox'
            cond_red_col = f'cond_{T}_red'
            
            if cond_ox_col in subset.columns and cond_red_col in subset.columns:
                ox_vals = subset[cond_ox_col].dropna()
                red_vals = subset[cond_red_col].dropna()
                
                if len(ox_vals) > 0 and len(red_vals) > 0:
                    ratio = ox_vals.mean() / red_vals.mean() if red_vals.mean() > 0 else np.nan
                    ratios.append(ratio)
                else:
                    ratios.append(np.nan)
            else:
                ratios.append(np.nan)
        
        if not np.all(np.isnan(ratios)):
            ax4.plot(temperatures[:len(ratios)], ratios, 'o-', linewidth=2, markersize=8,
                    label=dt, color=MODERN_COLORS.get(dt, MODERN_COLORS['gray']))
    
    ax4.set_xlabel('Temperature (°C)', fontsize=11)
    ax4.set_ylabel('Conductivity Ratio (Oxidizing / Reducing)', fontsize=11)
    ax4.set_title('Ox/Red Conductivity Ratio vs Temperature', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal conductivity')
    
    plt.suptitle('Temperature Dependence of Key Properties', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_optimization_3d_surface(df, weights):
    """Create 3D surface plot for multi-objective optimization"""
    
    # Calculate composite score
    df_copy = df.copy()
    df_copy['composite_score'] = df_copy.apply(lambda row: calculate_composite_score(row, weights), axis=1)
    
    # Select key variables for 3D plot
    x_var = 'total_B_doping'
    y_var = 'avg_EN_B'
    z_var = 'composite_score'
    
    # Filter valid data
    plot_data = df_copy[[x_var, y_var, z_var]].dropna()
    
    if len(plot_data) < 10:
        return None
    
    # Create grid for surface
    x_grid = np.linspace(plot_data[x_var].min(), plot_data[x_var].max(), 30)
    y_grid = np.linspace(plot_data[y_var].min(), plot_data[y_var].max(), 30)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate Z values
    try:
        Z = griddata((plot_data[x_var], plot_data[y_var]), plot_data[z_var], 
                    (X, Y), method='cubic')
        
        # Handle NaN values
        Z = np.nan_to_num(Z, nan=plot_data[z_var].min())
    except:
        return None
    
    # Create 3D surface plot
    fig = go.Figure(data=[
        go.Surface(x=x_grid, y=y_grid, z=Z, colorscale='Viridis',
                  contours=dict(z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project=dict(z=True))))
    ])
    
    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=plot_data[x_var], y=plot_data[y_var], z=plot_data[z_var],
        mode='markers',
        marker=dict(size=5, color='red', symbol='circle'),
        name='Experimental data'
    ))
    
    # Find and highlight optimum
    max_idx = plot_data[z_var].idxmax()
    max_point = plot_data.loc[max_idx]
    
    fig.add_trace(go.Scatter3d(
        x=[max_point[x_var]], y=[max_point[y_var]], z=[max_point[z_var]],
        mode='markers',
        marker=dict(size=12, color='gold', symbol='star', line=dict(width=2, color='black')),
        name='Optimal composition'
    ))
    
    fig.update_layout(
        title=f'Multi-objective Optimization Landscape<br>Composite Score = f(B-site doping, B electronegativity)',
        scene=dict(
            xaxis_title='Total B-site Doping Concentration',
            yaxis_title='Average B-site Electronegativity',
            zaxis_title='Composite Score (higher = better)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    return fig

# =============================================================================
# ML Model training for property prediction
# =============================================================================
@st.cache_resource
def train_prediction_models(df):
    """Train ML models for predicting key properties"""
    
    # Define feature columns
    feature_cols = [
        'x_A', 'total_B_doping', 'deficit_magnitude',
        'avg_EN_A', 'avg_EN_B', 'EN_diff_AB', 'EN_diff_BO',
        'avg_polarizability_A', 'avg_polarizability_B',
        'avg_IP_A', 'avg_IP_B', 'IP_diff_AB'
    ]
    
    # Define target columns
    target_cols = {
        'cond_700_ox': 'Conductivity at 700°C',
        'power_700': 'Power Density at 700°C',
        'ASR_700': 'ASR at 700°C',
        'TEC_avg': 'Average TEC'
    }
    
    # Filter available features
    available_features = [f for f in feature_cols if f in df.columns]
    
    if len(available_features) < 3:
        return None, None
    
    # Prepare data
    X = df[available_features].fillna(0)
    
    # Encode doping type as one-hot
    doping_dummies = pd.get_dummies(df['doping_type'], prefix='doping')
    X = pd.concat([X, doping_dummies], axis=1)
    
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    cv_scores = {}
    
    for target_col, target_name in target_cols.items():
        if target_col in df.columns:
            y = df[target_col].dropna()
            X_aligned = X_scaled[~df[target_col].isna()]
            
            if len(X_aligned) >= 10:
                # Train XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                
                # Cross-validation
                try:
                    cv_scores[target_col] = cross_val_score(model, X_aligned, y, 
                                                            cv=min(3, len(X_aligned)), 
                                                            scoring='r2', n_jobs=-1)
                    model.fit(X_aligned, y)
                    models[target_col] = model
                except Exception as e:
                    print(f"Could not train model for {target_col}: {e}")
    
    return {
        'models': models,
        'scaler': scaler,
        'feature_names': feature_names,
        'cv_scores': cv_scores
    }, X

# =============================================================================
# Streamlit UI Components
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
# Main Streamlit App
# =============================================================================
def main():
    st.set_page_config(
        page_title="BaFeO3 Electrode Analyzer v1.0",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_modern_styling()
    
    st.markdown("""
    <h1>
        ⚡ BaFeO₃ Electrode Materials Analyzer
        <span class="version-badge">v1.0</span>
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p style="font-size: 1.1rem; margin-bottom: 0;">
        Advanced platform for analyzing doped BaFeO₃-based electrode materials for SOFC applications.
        Compare doping strategies, optimize multiple properties simultaneously, and discover 
        composition-performance relationships.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Excel file with electrode data", 
        type=['xlsx', 'xls'],
        help="File should have headers in rows 0-1 and data starting from row 2"
    )
    
    if uploaded_file is None:
        st.info("Please upload an Excel file to begin analysis")
        return
    
    # Load data
    with st.spinner("Loading and processing data..."):
        df = load_excel_data(uploaded_file)
    
    if df is None or len(df) == 0:
        st.error("Failed to load data. Please check file format.")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        page = st.radio(
            "Select Module",
            ["📊 Data Overview", 
             "🏆 Doping Strategy Analysis",
             "🎯 Multi-Objective Optimization",
             "⚖️ Property Weighting",
             "🔬 Composition-Property Correlations",
             "📈 Temperature Dependence",
             "🤖 ML Predictor",
             "📊 Advanced Analytics",
             "ℹ️ About"]
        )
        
        st.markdown("---")
        
        # Dataset statistics
        st.markdown("## 📊 Dataset Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">{len(df)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Doping Strategies</div>
                <div class="metric-value">{df['doping_type'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filter options
        st.markdown("## 🔍 Filters")
        
        doping_types = ['All'] + sorted(df['doping_type'].unique().tolist())
        selected_doping = st.selectbox("Doping Strategy", doping_types)
        
        if selected_doping != 'All':
            df = df[df['doping_type'] == selected_doping]
    
    # =========================================================================
    # Page 1: Data Overview
    # =========================================================================
    if page == "📊 Data Overview":
        st.markdown("## 📊 Dataset Overview")
        
        # Display data table
        with st.expander("📋 Data Table", expanded=True):
            display_cols = ['Composition', 'doping_type', 'x_A', 'total_B_doping', 
                           'cond_700_ox', 'power_700', 'ASR_700', 'TEC_avg', 'doi']
            available_display = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_display], use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics by Doping Strategy")
        
        summary_stats = df.groupby('doping_type').agg({
            'cond_700_ox': ['count', 'mean', 'std', 'min', 'max'],
            'power_700': ['count', 'mean', 'std', 'min', 'max'],
            'ASR_700': ['count', 'mean', 'std', 'min', 'max'],
            'TEC_avg': ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        st.dataframe(summary_stats, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Filtered Data", csv, "baFeO3_data.csv", "text/csv")
    
    # =========================================================================
    # Page 2: Doping Strategy Analysis
    # =========================================================================
    elif page == "🏆 Doping Strategy Analysis":
        st.markdown("## 🏆 Doping Strategy Comparison")
        
        st.markdown("""
        <div class="card">
            <p>This section compares five doping strategies:</p>
            <ul>
                <li><b style="color:#2ecc71">A_only</b> - Only A-site doping (Ba replaced by Sr, La, Ca, etc.)</li>
                <li><b style="color:#e74c3c">B_only</b> - Only B-site doping (Fe replaced by Zn, Zr, Ce, Ni, etc.)</li>
                <li><b style="color:#3498db">AB</b> - Co-doping on both A and B sites</li>
                <li><b style="color:#f39c12">A_defect</b> - A-site deficiency (z < 1)</li>
                <li><b style="color:#9b59b6">B_defect</b> - B-site deficiency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Strategy comparison boxplots
        fig1 = create_doping_strategy_comparison(df)
        st.pyplot(fig1)
        plt.close()
        
        # Radar chart
        st.markdown("### 📊 Radar Chart: Strategy Performance Profile")
        fig2 = create_radar_chart_strategies(df)
        st.pyplot(fig2)
        plt.close()
        
        # Dopant concentration effects
        st.markdown("### 📈 Dopant Concentration Effects")
        fig3 = create_dopant_concentration_effects(df)
        st.pyplot(fig3)
        plt.close()
        
        # Best performers table
        st.markdown("### 🏆 Top Performers by Property")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Highest Conductivity at 700°C**")
            top_cond = df.nlargest(10, 'cond_700_ox')[['Composition', 'doping_type', 'cond_700_ox']].dropna()
            st.dataframe(top_cond, use_container_width=True)
        
        with col2:
            st.markdown("**Highest Power Density at 700°C**")
            top_power = df.nlargest(10, 'power_700')[['Composition', 'doping_type', 'power_700']].dropna()
            st.dataframe(top_power, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Lowest ASR at 700°C**")
            top_asr = df.nsmallest(10, 'ASR_700')[['Composition', 'doping_type', 'ASR_700']].dropna()
            st.dataframe(top_asr, use_container_width=True)
        
        with col2:
            st.markdown("**Lowest TEC (best match with electrolytes)**")
            top_tec = df.nsmallest(10, 'TEC_avg')[['Composition', 'doping_type', 'TEC_avg']].dropna()
            st.dataframe(top_tec, use_container_width=True)
    
    # =========================================================================
    # Page 3: Multi-Objective Optimization
    # =========================================================================
    elif page == "🎯 Multi-Objective Optimization":
        st.markdown("## 🎯 Multi-Objective Optimization")
        
        st.markdown("""
        <div class="card">
            <p>Define your optimization priorities using the sliders below. 
            The composite score combines all properties with specified weights.</p>
            <p><b>Higher score = better overall performance</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Weight sliders
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Properties to Maximize")
            weight_conductivity = st.slider("Conductivity (higher is better)", 0.0, 1.0, 0.4, 0.05)
            weight_power = st.slider("Power Density (higher is better)", 0.0, 1.0, 0.4, 0.05)
        
        with col2:
            st.markdown("### Properties to Minimize")
            weight_ASR = st.slider("ASR (lower is better)", 0.0, 1.0, 0.3, 0.05)
            weight_TEC = st.slider("TEC (lower is better)", 0.0, 1.0, 0.2, 0.05)
            weight_TEC_stability = st.slider("TEC Stability (small ΔTEC)", 0.0, 1.0, 0.1, 0.05)
        
        weights = {
            'conductivity': weight_conductivity,
            'power': weight_power,
            'ASR': weight_ASR,
            'TEC': weight_TEC,
            'TEC_stability': weight_TEC_stability
        }
        
        # Calculate composite scores
        df['composite_score'] = df.apply(lambda row: calculate_composite_score(row, weights), axis=1)
        
        # Display top compositions
        st.markdown("### 🏆 Top 10 Optimal Compositions")
        top_compositions = get_optimal_compositions(df, weights, top_n=10)
        
        display_cols = ['Composition', 'doping_type', 'cond_700_ox', 'power_700', 'ASR_700', 'TEC_avg', 'composite_score']
        available_display = [c for c in display_cols if c in top_compositions.columns]
        
        st.dataframe(top_compositions[available_display], use_container_width=True)
        
        # Parallel coordinates
        st.markdown("### 📊 Multi-Dimensional Performance Visualization")
        fig1 = create_parallel_coordinates_optimization(df, weights)
        if fig1:
            st.pyplot(fig1)
            plt.close()
        
        # 3D optimization landscape
        st.markdown("### 🌄 3D Optimization Landscape")
        fig2 = create_optimization_3d_surface(df, weights)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Insufficient data for 3D surface plot")
        
        # Strategy effectiveness
        st.markdown("### 📊 Strategy Effectiveness by Weighted Score")
        
        strategy_scores = df.groupby('doping_type')['composite_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(strategy_scores.index, strategy_scores['mean'], 
                     yerr=strategy_scores['std'], capsize=5,
                     color=[MODERN_COLORS.get(dt, MODERN_COLORS['gray']) for dt in strategy_scores.index],
                     edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Doping Strategy', fontsize=11)
        ax.set_ylabel('Average Composite Score', fontsize=11)
        ax.set_title('Doping Strategy Effectiveness with Current Weights', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, strategy_scores['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{score:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # =========================================================================
    # Page 4: Property Weighting
    # =========================================================================
    elif page == "⚖️ Property Weighting":
        st.markdown("## ⚖️ Interactive Property Weighting Explorer")
        
        st.markdown("""
        <div class="card">
            <p>Adjust weights to see how different optimization priorities affect material ranking.
            This helps identify which doping strategy works best for specific applications.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for weight controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Maximize")
            w_cond = st.slider("⚡ Conductivity", 0, 100, 40, 5, key="w_cond")
            w_power = st.slider("🔋 Power Density", 0, 100, 40, 5, key="w_power")
        
        with col2:
            st.markdown("### Minimize")
            w_asr = st.slider("📉 ASR", 0, 100, 30, 5, key="w_asr")
            w_tec = st.slider("📏 TEC", 0, 100, 20, 5, key="w_tec")
            w_tec_stab = st.slider("⚖️ TEC Stability", 0, 100, 10, 5, key="w_tec_stab")
        
        # Normalize weights
        total = w_cond + w_power + w_asr + w_tec + w_tec_stab
        if total > 0:
            weights = {
                'conductivity': w_cond / total,
                'power': w_power / total,
                'ASR': w_asr / total,
                'TEC': w_tec / total,
                'TEC_stability': w_tec_stab / total
            }
        else:
            weights = {'conductivity': 0, 'power': 0, 'ASR': 0, 'TEC': 0, 'TEC_stability': 0}
        
        # Calculate scores
        df['composite_score'] = df.apply(lambda row: calculate_composite_score(row, weights), axis=1)
        
        # Display current weighting scheme
        st.markdown("### Current Weighting Scheme")
        
        weight_df = pd.DataFrame({
            'Property': ['Conductivity', 'Power Density', 'ASR', 'TEC', 'TEC Stability'],
            'Direction': ['Maximize', 'Maximize', 'Minimize', 'Minimize', 'Minimize'],
            'Weight': [weights['conductivity'], weights['power'], weights['ASR'], weights['TEC'], weights['TEC_stability']],
            'Weight %': [f"{weights['conductivity']*100:.1f}%", f"{weights['power']*100:.1f}%", 
                        f"{weights['ASR']*100:.1f}%", f"{weights['TEC']*100:.1f}%", 
                        f"{weights['TEC_stability']*100:.1f}%"]
        })
        
        st.dataframe(weight_df, use_container_width=True)
        
        # Top compositions with current weights
        st.markdown("### 🏆 Top 10 Compositions with Current Weights")
        top_df = get_optimal_compositions(df, weights, top_n=10)
        
        display_cols = ['Composition', 'doping_type', 'cond_700_ox', 'power_700', 'ASR_700', 'TEC_avg', 'composite_score']
        available_display = [c for c in display_cols if c in top_df.columns]
        st.dataframe(top_df[available_display], use_container_width=True)
        
        # Strategy performance under current weights
        st.markdown("### Strategy Performance Under Current Weights")
        
        strategy_performance = df.groupby('doping_type')['composite_score'].agg(['mean', 'std', 'count', 'max']).sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(strategy_performance))
        bars = ax.bar(x_pos, strategy_performance['mean'], yerr=strategy_performance['std'],
                     capsize=5, alpha=0.7, edgecolor='black',
                     color=[MODERN_COLORS.get(dt, MODERN_COLORS['gray']) for dt in strategy_performance.index])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategy_performance.index, rotation=45, ha='right')
        ax.set_ylabel('Average Composite Score', fontsize=11)
        ax.set_title('Doping Strategy Ranking with Current Weights', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, strategy_performance['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Interactive comparison
        st.markdown("### 🔍 Compare Specific Strategies")
        
        selected_strategies = st.multiselect(
            "Select strategies to compare",
            df['doping_type'].unique(),
            default=list(df['doping_type'].unique())[:3]
        )
        
        if selected_strategies:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for strategy in selected_strategies:
                subset = df[df['doping_type'] == strategy]['composite_score'].dropna()
                if len(subset) > 0:
                    ax.hist(subset, bins=15, alpha=0.5, label=strategy, 
                           edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Composite Score', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Composite Scores by Strategy', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
    
    # =========================================================================
    # Page 5: Composition-Property Correlations
    # =========================================================================
    elif page == "🔬 Composition-Property Correlations":
        st.markdown("## 🔬 Composition-Property Correlation Analysis")
        
        # Correlation heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        fig1 = create_correlation_heatmap(df)
        if fig1:
            st.pyplot(fig1)
            plt.close()
        
        # Pairplot of key variables
        st.markdown("### 📊 Pairplot of Key Variables")
        
        plot_vars = ['total_B_doping', 'avg_EN_B', 'cond_700_ox', 'power_700', 'ASR_700', 'TEC_avg']
        available_vars = [v for v in plot_vars if v in df.columns]
        
        if len(available_vars) >= 2:
            # Create pairplot using seaborn
            fig, axes = plt.subplots(len(available_vars), len(available_vars), 
                                    figsize=(15, 15))
            
            for i, var_i in enumerate(available_vars):
                for j, var_j in enumerate(available_vars):
                    ax = axes[i, j]
                    
                    if i == j:
                        # Histogram on diagonal
                        ax.hist(df[var_i].dropna(), bins=20, color=MODERN_COLORS['primary'],
                               edgecolor='black', alpha=0.7)
                        ax.set_xlabel(var_i.replace('_', ' ').title(), fontsize=8)
                    else:
                        # Scatter plot
                        scatter_data = df[[var_i, var_j]].dropna()
                        if len(scatter_data) > 0:
                            ax.scatter(scatter_data[var_i], scatter_data[var_j], 
                                      alpha=0.6, s=30, c=MODERN_COLORS['secondary'],
                                      edgecolors='black', linewidth=0.3)
                            
                            # Add trend line
                            if len(scatter_data) > 2:
                                z = np.polyfit(scatter_data[var_i], scatter_data[var_j], 1)
                                x_line = np.linspace(scatter_data[var_i].min(), scatter_data[var_i].max(), 50)
                                ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5, linewidth=1)
                                
                                # Add correlation
                                corr = scatter_data[var_i].corr(scatter_data[var_j])
                                ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                                       fontsize=8, fontweight='bold',
                                       bbox=dict(facecolor='white', alpha=0.8))
                        
                        ax.set_xlabel(var_i.replace('_', ' ').title(), fontsize=8)
                    
                    if i == len(available_vars) - 1:
                        ax.set_xlabel(var_j.replace('_', ' ').title(), fontsize=8)
                    if j == 0:
                        ax.set_ylabel(var_i.replace('_', ' ').title(), fontsize=8)
                    
                    ax.tick_params(axis='both', labelsize=7)
            
            plt.suptitle('Pairplot of Key Variables', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Electronegativity vs Properties
        st.markdown("### ⚡ Electronegativity Effects")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Scatter with color by doping type
            for dt in df['doping_type'].unique():
                subset = df[df['doping_type'] == dt].dropna(subset=['avg_EN_B', 'cond_700_ox'])
                if len(subset) > 0:
                    ax.scatter(subset['avg_EN_B'], subset['cond_700_ox'], 
                              label=dt, s=60, alpha=0.7,
                              edgecolors='black', linewidth=0.5,
                              color=MODERN_COLORS.get(dt, MODERN_COLORS['gray']))
            
            ax.set_xlabel('Average B-site Electronegativity', fontsize=11)
            ax.set_ylabel('Conductivity at 700°C (S/cm)', fontsize=11)
            ax.set_title('EN_B vs Conductivity', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for dt in df['doping_type'].unique():
                subset = df[df['doping_type'] == dt].dropna(subset=['avg_EN_B', 'power_700'])
                if len(subset) > 0:
                    ax.scatter(subset['avg_EN_B'], subset['power_700'], 
                              label=dt, s=60, alpha=0.7,
                              edgecolors='black', linewidth=0.5,
                              color=MODERN_COLORS.get(dt, MODERN_COLORS['gray']))
            
            ax.set_xlabel('Average B-site Electronegativity', fontsize=11)
            ax.set_ylabel('Power Density at 700°C (mW/cm²)', fontsize=11)
            ax.set_title('EN_B vs Power Density', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # =========================================================================
    # Page 6: Temperature Dependence
    # =========================================================================
    elif page == "📈 Temperature Dependence":
        st.markdown("## 📈 Temperature Dependence of Properties")
        
        fig = create_temperature_dependence_plots(df)
        st.pyplot(fig)
        plt.close()
        
        # Activation energy calculation
        st.markdown("### ⚡ Activation Energy Analysis")
        
        st.markdown("""
        <div class="card">
            <p>Activation energy (Ea) for conductivity is calculated from Arrhenius equation:</p>
            <p><b>σ = σ₀·exp(-Ea/kT)</b> → <b>ln(σ) = ln(σ₀) - Ea/(k·T)</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate activation energies for each composition
        activation_data = []
        
        for idx, row in df.iterrows():
            T_K = [500+273.15, 600+273.15, 700+273.15]
            cond_vals = [row.get('cond_500_ox'), row.get('cond_600_ox'), row.get('cond_700_ox')]
            
            # Filter valid data points
            valid_pairs = [(T, cond) for T, cond in zip(T_K, cond_vals) if not pd.isna(cond) and cond > 0]
            
            if len(valid_pairs) >= 2:
                T_arr = np.array([1/(p[0]) for p in valid_pairs])
                ln_sigma = np.array([np.log(p[1]) for p in valid_pairs])
                
                # Linear fit
                z = np.polyfit(T_arr, ln_sigma, 1)
                Ea = -z[0] * 8.314e-3  # Convert to kJ/mol
                R2 = np.corrcoef(T_arr, ln_sigma)[0,1]**2
                
                activation_data.append({
                    'Composition': row.get('Composition', f'Entry_{idx}'),
                    'doping_type': row.get('doping_type', 'unknown'),
                    'Ea_ox (kJ/mol)': Ea,
                    'R2': R2,
                    'n_points': len(valid_pairs)
                })
        
        if activation_data:
            df_ea = pd.DataFrame(activation_data)
            
            # Display activation energies
            st.dataframe(df_ea, use_container_width=True)
            
            # Boxplot by doping strategy
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_to_plot = []
            for dt in df_ea['doping_type'].unique():
                data = df_ea[df_ea['doping_type'] == dt]['Ea_ox (kJ/mol)'].dropna()
                if len(data) > 0:
                    data_to_plot.append(data.values)
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=df_ea['doping_type'].unique(), patch_artist=True)
                colors = [MODERN_COLORS.get(dt, MODERN_COLORS['gray']) for dt in df_ea['doping_type'].unique()]
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_xlabel('Doping Strategy', fontsize=11)
                ax.set_ylabel('Activation Energy Ea (kJ/mol)', fontsize=11)
                ax.set_title('Activation Energy for Conductivity by Doping Strategy', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                st.pyplot(fig)
                plt.close()
    
    # =========================================================================
    # Page 7: ML Predictor
    # =========================================================================
    elif page == "🤖 ML Predictor":
        st.markdown("## 🤖 Machine Learning Property Predictor")
        
        model_data, X_train = train_prediction_models(df)
        
        if model_data is None or len(model_data['models']) == 0:
            st.warning("Insufficient data for ML model training. Need at least 10 samples with complete data.")
            return
        
        st.markdown("""
        <div class="card">
            <p>Enter composition parameters to predict material properties using ensemble ML models.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_A = st.slider("A-site Doping Concentration (x)", 0.0, 0.8, 0.2, 0.01)
            total_B_doping = st.slider("Total B-site Doping Concentration", 0.0, 0.8, 0.2, 0.01)
        
        with col2:
            avg_EN_B = st.slider("Average B-site Electronegativity", 1.5, 2.5, 1.9, 0.01)
            EN_diff_BO = st.slider("B-O Electronegativity Difference", 1.5, 2.5, 2.0, 0.01)
        
        with col3:
            deficit_magnitude = st.slider("Deficit Magnitude (1-z)", 0.0, 0.3, 0.0, 0.01)
            doping_type = st.selectbox("Doping Strategy", ['B_only', 'A_only', 'AB', 'undoped'])
        
        # Prepare feature vector
        feature_vector = {
            'x_A': x_A,
            'total_B_doping': total_B_doping,
            'deficit_magnitude': deficit_magnitude,
            'avg_EN_B': avg_EN_B,
            'EN_diff_BO': EN_diff_BO,
            'avg_EN_A': get_electronegativity('Ba') if doping_type == 'A_only' else get_electronegativity('Ba'),
            'avg_polarizability_B': 0.18,  # Default Fe polarizability
            'avg_IP_B': 7.90,  # Default Fe IP
        }
        
        # Add one-hot encoding for doping type
        for dt in ['doping_A_only', 'doping_B_only', 'doping_AB', 'doping_undoped']:
            feature_vector[dt] = 1 if dt == f'doping_{doping_type}' else 0
        
        # Create DataFrame
        X_pred = pd.DataFrame([feature_vector])
        
        # Ensure all features are present
        for col in model_data['feature_names']:
            if col not in X_pred.columns:
                X_pred[col] = 0
        
        X_pred = X_pred[model_data['feature_names']]
        X_pred_scaled = model_data['scaler'].transform(X_pred)
        
        # Make predictions
        st.markdown("### 📊 Predicted Properties")
        
        col1, col2, col3, col4 = st.columns(4)
        
        predictions = {}
        for target_col, model in model_data['models'].items():
            pred = model.predict(X_pred_scaled)[0]
            predictions[target_col] = pred
            
            if target_col == 'cond_700_ox':
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Conductivity at 700°C</div>
                        <div class="metric-value">{pred:.1f}</div>
                        <div class="metric-delta">S/cm</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif target_col == 'power_700':
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Power Density at 700°C</div>
                        <div class="metric-value">{pred:.0f}</div>
                        <div class="metric-delta">mW/cm²</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif target_col == 'ASR_700':
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">ASR at 700°C</div>
                        <div class="metric-value">{pred:.2f}</div>
                        <div class="metric-delta">Ω·cm²</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif target_col == 'TEC_avg':
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Average TEC</div>
                        <div class="metric-value">{pred:.1f}</div>
                        <div class="metric-delta">×10⁻⁶ K⁻¹</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Model performance
        with st.expander("📊 Model Performance Metrics"):
            for target_col, scores in model_data['cv_scores'].items():
                st.metric(f"{target_col} - R² (CV)", 
                         f"{scores.mean():.3f} ± {scores.std():.3f}")
    
    # =========================================================================
    # Page 8: Advanced Analytics
    # =========================================================================
    elif page == "📊 Advanced Analytics":
        st.markdown("## 📊 Advanced Statistical Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Outlier Detection", "PCA Analysis"])
        
        with tab1:
            st.markdown("### Property Distributions")
            
            prop_to_plot = st.selectbox("Select property", 
                                       ['cond_700_ox', 'power_700', 'ASR_700', 'TEC_avg'])
            
            if prop_to_plot in df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histogram
                axes[0].hist(df[prop_to_plot].dropna(), bins=20, color=MODERN_COLORS['primary'],
                            edgecolor='black', alpha=0.7)
                axes[0].set_xlabel(prop_to_plot.replace('_', ' ').title(), fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].set_title('Histogram', fontsize=12, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                # Q-Q plot
                from scipy import stats
                stats.probplot(df[prop_to_plot].dropna(), dist="norm", plot=axes[1])
                axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Statistics
                st.markdown("### Descriptive Statistics")
                stats_df = df[prop_to_plot].describe()
                stats_df['skewness'] = df[prop_to_plot].skew()
                stats_df['kurtosis'] = df[prop_to_plot].kurtosis()
                st.dataframe(pd.DataFrame(stats_df).T, use_container_width=True)
        
        with tab2:
            st.markdown("### Outlier Detection")
            
            prop_for_outliers = st.selectbox("Select property for outlier detection",
                                            ['cond_700_ox', 'power_700', 'ASR_700', 'TEC_avg'],
                                            key="outlier_prop")
            
            method = st.selectbox("Detection method", ["IQR", "Z-score", "Modified Z-score"])
            
            if prop_for_outliers in df.columns:
                data = df[prop_for_outliers].dropna()
                
                if method == "IQR":
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    threshold = 1.5
                    outliers = data[(data < Q1 - threshold * IQR) | (data > Q3 + threshold * IQR)]
                elif method == "Z-score":
                    z_scores = np.abs(stats.zscore(data))
                    threshold = 3
                    outliers = data[z_scores > threshold]
                else:  # Modified Z-score
                    median = data.median()
                    mad = stats.median_abs_deviation(data)
                    modified_z_scores = 0.6745 * (data - median) / mad
                    threshold = 3.5
                    outliers = data[np.abs(modified_z_scores) > threshold]
                
                st.metric("Number of outliers detected", len(outliers))
                st.metric("Percentage", f"{len(outliers)/len(data)*100:.1f}%")
                
                if len(outliers) > 0:
                    st.markdown("### Outlier Compositions")
                    outlier_indices = outliers.index
                    outlier_data = df.loc[outlier_indices, ['Composition', 'doping_type', prop_for_outliers, 'doi']]
                    st.dataframe(outlier_data, use_container_width=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Boxplot
                bp = ax.boxplot(data, patch_artist=True)
                bp['boxes'][0].set_facecolor(MODERN_COLORS['primary'])
                bp['boxes'][0].set_alpha(0.7)
                
                # Highlight outliers
                if len(outliers) > 0:
                    ax.plot(np.ones_like(outliers.values), outliers.values, 'ro', 
                           markersize=8, label=f'Outliers (n={len(outliers)})')
                
                ax.set_xticklabels([prop_for_outliers.replace('_', ' ').title()])
                ax.set_ylabel(prop_for_outliers.replace('_', ' ').title(), fontsize=11)
                ax.set_title(f'Outlier Detection: {method}', fontsize=12, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
        
        with tab3:
            st.markdown("### Principal Component Analysis (PCA)")
            
            # Select features for PCA
            pca_features = ['x_A', 'total_B_doping', 'avg_EN_B', 'EN_diff_BO', 
                           'cond_700_ox', 'power_700', 'ASR_700', 'TEC_avg']
            
            available_pca = [f for f in pca_features if f in df.columns]
            
            if len(available_pca) >= 3:
                # Prepare data
                pca_data = df[available_pca].dropna()
                
                if len(pca_data) >= 5:
                    # Perform PCA
                    scaler = StandardScaler()
                    pca_scaled = scaler.fit_transform(pca_data)
                    pca = PCA(n_components=min(3, len(available_pca)))
                    pca_result = pca.fit_transform(pca_scaled)
                    
                    # Explained variance
                    fig, ax = plt.subplots(figsize=(10, 5))
                    explained_var = pca.explained_variance_ratio_
                    ax.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7,
                          color=MODERN_COLORS['primary'], edgecolor='black')
                    ax.plot(range(1, len(explained_var)+1), np.cumsum(explained_var), 
                           'ro-', linewidth=2, markersize=8, label='Cumulative')
                    ax.set_xlabel('Principal Component', fontsize=11)
                    ax.set_ylabel('Explained Variance Ratio', fontsize=11)
                    ax.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    
                    # 2D PCA plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    for dt in pca_data['doping_type'].unique():
                        mask = pca_data['doping_type'] == dt
                        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                                  label=dt, s=80, alpha=0.7,
                                  edgecolors='black', linewidth=0.5,
                                  color=MODERN_COLORS.get(dt, MODERN_COLORS['gray']))
                    
                    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=11)
                    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=11)
                    ax.set_title('PCA Projection: Materials Clustering', fontsize=12, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Feature loadings
                    st.markdown("### PCA Loadings (Feature Contributions)")
                    loadings_df = pd.DataFrame(pca.components_[:2].T,
                                              columns=['PC1', 'PC2'],
                                              index=available_pca)
                    st.dataframe(loadings_df, use_container_width=True)
    
    # =========================================================================
    # Page 9: About
    # =========================================================================
    else:
        st.markdown("## ℹ️ About BaFeO₃ Electrode Analyzer")
        
        st.markdown("""
        <div class="card">
            <h3>Advanced Tool for SOFC Electrode Materials</h3>
            <p>Version 1.0 is specifically designed for analyzing doped BaFeO₃-based perovskite 
            electrode materials for Solid Oxide Fuel Cells (SOFCs).</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>🎯 Key Features</h4>
                <ul>
                    <li><b>Doping Strategy Analysis</b>: Compare A-site, B-site, co-doping, and defect strategies</li>
                    <li><b>Multi-Objective Optimization</b>: Weighted scoring of multiple properties</li>
                    <li><b>Electronic Descriptors</b>: Electronegativity, polarizability, ionization potential</li>
                    <li><b>Temperature Dependence</b>: Conductivity, power, ASR vs T</li>
                    <li><b>ML Predictions</b>: Property prediction using ensemble models</li>
                    <li><b>Correlation Analysis</b>: Identify key composition-property relationships</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>📊 Key Properties Analyzed</h4>
                <ul>
                    <li><b>Conductivity</b>: 500-700°C in oxidizing/reducing atmospheres</li>
                    <li><b>Power Density</b>: 600-700°C (mW/cm²)</li>
                    <li><b>ASR</b>: Area Specific Resistance (Ω·cm²)</li>
                    <li><b>TEC</b>: Thermal Expansion Coefficient (×10⁻⁶ K⁻¹)</li>
                    <li><b>TEC Stability</b>: Δ between LT and HT regions</li>
                </ul>
            </div>
            
            <div class="card">
                <h4>📝 Data Format</h4>
                <p>Excel file with:</p>
                <ul>
                    <li>Row 0: Parameter categories</li>
                    <li>Row 1: Detailed column descriptions</li>
                    <li>Row 2+: Experimental data</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="footer">
            <p>© 2025 BaFeO₃ Electrode Materials Analyzer | Developed for SOFC Materials Research</p>
            <p>For questions, suggestions, or data contributions, please contact the developers</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Run the app
# =============================================================================
if __name__ == "__main__":
    main()


from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

def standardize_data(df, features):
    """Standardize features to z-scores"""
    standardized = {}
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        standardized[feature] = {
            'z_scores': ((df[feature] - mean) / std).tolist(),
            'mean': mean,
            'std': std
        }
    return standardized

def compute_covariance_matrix(z_scores_dict, n):
    """Compute covariance matrix from z-scores"""
    features = list(z_scores_dict.keys())
    k = len(features)
    cov_matrix = np.zeros((k, k))
    
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            z1 = np.array(z_scores_dict[f1]['z_scores'])
            z2 = np.array(z_scores_dict[f2]['z_scores'])
            cov_matrix[i, j] = np.sum(z1 * z2) / (n - 1)
    
    return cov_matrix

def perform_pca(df, features):
    """Perform complete PCA analysis"""
    n = len(df)
    
    # Step 1: Standardize
    standardized = standardize_data(df, features)
    
    # Step 2: Covariance matrix
    cov_matrix = compute_covariance_matrix(standardized, n)
    
    # Step 3: Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 4: Compute PC scores
    z_matrix = np.column_stack([standardized[f]['z_scores'] for f in features])
    pc_scores = np.dot(z_matrix, eigenvectors)
    
    # Calculate variance explained
    total_variance = np.sum(eigenvalues)
    variance_explained = (eigenvalues / total_variance * 100).tolist()
    cumulative_variance = np.cumsum(variance_explained).tolist()
    
    # Get unique species and assign consistent indices
    species_list = df['species'].tolist() if 'species' in df.columns else None
    
    return {
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'pc_scores': pc_scores.tolist(),
        'variance_explained': variance_explained,
        'cumulative_variance': cumulative_variance,
        'features': features,
        'standardized': standardized,
        'species': species_list
    }

@app.route('/')
def index():
    return render_template('pca_index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Load iris dataset (you should have this CSV file)
        df = pd.read_csv('iris.csv')
        
        # Perform PCA on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = perform_pca(df, numeric_cols)
        result['success'] = True
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
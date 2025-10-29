"""
Predictive Surface Mapping
Creates multi-dimensional predictive surfaces for market analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.interpolate import griddata
import logging


class PredictiveSurface:
    """
    Advanced predictive surface mapping system
    Creates 3D predictive surfaces for market forecasting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('PredictiveSurface')
        self.config = config or {}
        
        # Predictive parameters
        self.predictive_horizon = self.config.get('predictive_horizon', 500)
        self.surface_resolution = 50  # Grid resolution for surface
        self.confidence_levels = [0.68, 0.95, 0.997]  # 1, 2, 3 sigma
        
        # Storage for surfaces
        self.surfaces: Dict[str, Dict] = {}
        
        # Gaussian Process model
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
    def create_price_surface(self, 
                            symbol: str,
                            price_data: pd.DataFrame,
                            features: List[str]) -> Dict:
        """
        Create predictive price surface
        
        Args:
            symbol: Trading symbol
            price_data: DataFrame with price and features
            features: List of feature column names
            
        Returns:
            Dictionary with surface data
        """
        if len(features) < 2:
            self.logger.error("Need at least 2 features for surface mapping")
            return {}
            
        # Use first two features for 2D surface
        X = price_data[features[:2]].values
        y = price_data['close'].values if 'close' in price_data.columns else price_data.iloc[:, 0].values
        
        # Fit Gaussian Process
        try:
            self.gp_model.fit(X, y)
        except Exception as e:
            self.logger.error(f"Error fitting GP model: {e}")
            return {}
            
        # Create prediction grid
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        
        x1_range = np.linspace(x1_min, x1_max, self.surface_resolution)
        x2_range = np.linspace(x2_min, x2_max, self.surface_resolution)
        
        X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
        X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
        
        # Make predictions
        y_pred, y_std = self.gp_model.predict(X_grid, return_std=True)
        
        # Reshape predictions to grid
        Z_pred = y_pred.reshape(X1_grid.shape)
        Z_std = y_std.reshape(X1_grid.shape)
        
        surface_data = {
            'symbol': symbol,
            'features': features[:2],
            'x1_grid': X1_grid,
            'x2_grid': X2_grid,
            'predictions': Z_pred,
            'std_dev': Z_std,
            'confidence_bounds': self._calculate_confidence_bounds(Z_pred, Z_std),
            'gradient': self._calculate_surface_gradient(Z_pred),
            'curvature': self._calculate_surface_curvature(Z_pred)
        }
        
        self.surfaces[symbol] = surface_data
        
        return surface_data
        
    def _calculate_confidence_bounds(self, predictions: np.ndarray, std_dev: np.ndarray) -> Dict:
        """
        Calculate confidence bounds for predictions
        
        Args:
            predictions: Predicted values
            std_dev: Standard deviation of predictions
            
        Returns:
            Dictionary with confidence bounds
        """
        bounds = {}
        
        for level in self.confidence_levels:
            # Calculate z-score for confidence level
            from scipy.stats import norm
            z_score = norm.ppf((1 + level) / 2)
            
            bounds[f'{int(level*100)}%'] = {
                'upper': predictions + z_score * std_dev,
                'lower': predictions - z_score * std_dev,
                'width': 2 * z_score * std_dev
            }
            
        return bounds
        
    def _calculate_surface_gradient(self, surface: np.ndarray) -> Dict:
        """
        Calculate gradient of predictive surface
        
        Args:
            surface: 2D array of predictions
            
        Returns:
            Dictionary with gradient information
        """
        # Calculate gradients in both directions
        grad_x = np.gradient(surface, axis=1)
        grad_y = np.gradient(surface, axis=0)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'dx': grad_x,
            'dy': grad_y,
            'magnitude': gradient_magnitude,
            'max_gradient': np.max(gradient_magnitude),
            'mean_gradient': np.mean(gradient_magnitude)
        }
        
    def _calculate_surface_curvature(self, surface: np.ndarray) -> Dict:
        """
        Calculate curvature of predictive surface
        
        Args:
            surface: 2D array of predictions
            
        Returns:
            Dictionary with curvature information
        """
        # Calculate second derivatives
        d2_x = np.gradient(np.gradient(surface, axis=1), axis=1)
        d2_y = np.gradient(np.gradient(surface, axis=0), axis=0)
        d2_xy = np.gradient(np.gradient(surface, axis=1), axis=0)
        
        # Calculate mean curvature
        mean_curvature = (d2_x + d2_y) / 2
        
        # Calculate Gaussian curvature
        gaussian_curvature = d2_x * d2_y - d2_xy**2
        
        return {
            'd2x': d2_x,
            'd2y': d2_y,
            'd2xy': d2_xy,
            'mean_curvature': mean_curvature,
            'gaussian_curvature': gaussian_curvature,
            'max_curvature': np.max(np.abs(mean_curvature))
        }
        
    def find_optimal_regions(self, symbol: str, optimization: str = 'max') -> List[Tuple]:
        """
        Find optimal regions on predictive surface
        
        Args:
            symbol: Trading symbol
            optimization: 'max' for maxima, 'min' for minima
            
        Returns:
            List of (x1, x2, value) tuples for optimal points
        """
        if symbol not in self.surfaces:
            self.logger.warning(f"No surface data for {symbol}")
            return []
            
        surface_data = self.surfaces[symbol]
        predictions = surface_data['predictions']
        x1_grid = surface_data['x1_grid']
        x2_grid = surface_data['x2_grid']
        
        # Find local extrema
        from scipy.ndimage import maximum_filter, minimum_filter
        
        if optimization == 'max':
            local_extrema = (predictions == maximum_filter(predictions, size=3))
        else:
            local_extrema = (predictions == minimum_filter(predictions, size=3))
            
        # Get coordinates of extrema
        extrema_coords = np.where(local_extrema)
        
        optimal_points = []
        for i, j in zip(extrema_coords[0], extrema_coords[1]):
            x1 = x1_grid[i, j]
            x2 = x2_grid[i, j]
            value = predictions[i, j]
            optimal_points.append((x1, x2, value))
            
        # Sort by value
        optimal_points.sort(key=lambda x: x[2], reverse=(optimization == 'max'))
        
        return optimal_points[:10]  # Return top 10
        
    def predict_trajectory(self, 
                          symbol: str,
                          current_state: np.ndarray,
                          steps: int = 10) -> Dict:
        """
        Predict future trajectory on surface
        
        Args:
            symbol: Trading symbol
            current_state: Current state [feature1, feature2]
            steps: Number of steps to predict
            
        Returns:
            Dictionary with trajectory prediction
        """
        if symbol not in self.surfaces:
            self.logger.warning(f"No surface data for {symbol}")
            return {}
            
        surface_data = self.surfaces[symbol]
        gradient = surface_data['gradient']
        
        # Initialize trajectory
        trajectory = [current_state.copy()]
        state = current_state.copy()
        
        # Follow gradient for specified steps
        step_size = 0.1  # Adaptive step size
        
        for _ in range(steps):
            # Get gradient at current position
            # Interpolate gradient from grid
            grad_x_interp = griddata(
                (surface_data['x1_grid'].ravel(), surface_data['x2_grid'].ravel()),
                gradient['dx'].ravel(),
                state.reshape(1, -1),
                method='linear'
            )[0]
            
            grad_y_interp = griddata(
                (surface_data['x1_grid'].ravel(), surface_data['x2_grid'].ravel()),
                gradient['dy'].ravel(),
                state.reshape(1, -1),
                method='linear'
            )[0]
            
            # Update state following gradient
            state = state + step_size * np.array([grad_x_interp, grad_y_interp])
            trajectory.append(state.copy())
            
        trajectory = np.array(trajectory)
        
        # Predict values along trajectory
        trajectory_predictions = []
        for point in trajectory:
            pred, std = self.gp_model.predict(point.reshape(1, -1), return_std=True)
            trajectory_predictions.append({
                'value': pred[0],
                'std': std[0]
            })
            
        return {
            'trajectory': trajectory,
            'predictions': trajectory_predictions,
            'direction': trajectory[-1] - trajectory[0],
            'distance': np.linalg.norm(trajectory[-1] - trajectory[0])
        }
        
    def get_uncertainty_map(self, symbol: str) -> Optional[np.ndarray]:
        """
        Get uncertainty map for predictions
        
        Args:
            symbol: Trading symbol
            
        Returns:
            2D array of prediction uncertainties
        """
        if symbol not in self.surfaces:
            return None
            
        return self.surfaces[symbol]['std_dev']
        
    def analyze_surface_stability(self, symbol: str) -> Dict:
        """
        Analyze stability characteristics of predictive surface
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with stability metrics
        """
        if symbol not in self.surfaces:
            return {}
            
        surface_data = self.surfaces[symbol]
        gradient = surface_data['gradient']
        curvature = surface_data['curvature']
        
        # Calculate stability metrics
        stability = {
            'gradient_stability': 1.0 / (1.0 + gradient['mean_gradient']),
            'curvature_stability': 1.0 / (1.0 + np.mean(np.abs(curvature['mean_curvature']))),
            'uncertainty_mean': np.mean(surface_data['std_dev']),
            'uncertainty_max': np.max(surface_data['std_dev']),
            'stable_regions': np.sum(gradient['magnitude'] < np.percentile(gradient['magnitude'], 25)),
            'volatile_regions': np.sum(gradient['magnitude'] > np.percentile(gradient['magnitude'], 75))
        }
        
        # Overall stability score (0-1, higher is more stable)
        stability['overall_score'] = (
            0.4 * stability['gradient_stability'] +
            0.3 * stability['curvature_stability'] +
            0.3 * (1.0 / (1.0 + stability['uncertainty_mean']))
        )
        
        return stability

# 3D Correlation Contagion Shock Simulator

## Creator/Dev
**tubakhxn**

## Project Overview
This project is a Streamlit application that simulates how correlation matrices break down during financial stress events and impact portfolio drawdowns. It visualizes the effects of contagion shocks on asset correlations, eigenvalue instability, and diversification collapse, helping users understand risk concentration and portfolio vulnerability.

## Features
- Interactive controls for asset count, base correlation, shock intensity, volatility, portfolio weights, and time steps
- 3D surface visualization of portfolio drawdown vs. shock and correlation
- Eigenvalue instability and diversification collapse metrics
- Fully modular, stable, and positive semi-definite matrix simulation
- Clean dark UI layout

## How to Fork
1. Click the "Fork" button at the top right of this repository on GitHub.
2. Clone your forked repository:
   ```
   git clone https://github.com/<your-username>/<your-forked-repo>.git
   ```
3. Install dependencies:
   ```
   pip install streamlit numpy pandas plotly scipy
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

---
For questions or contributions, contact tubakhxn.

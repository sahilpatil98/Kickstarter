# Kickstarter
Public code for my dissertation chapter:
A Kick or a start? Gender disparity evidence of winning a Kickstarter on future funding


There are three files:
1. Data Cleaner
- Cleans the raw dataset made available by WebRobots.io (https://webrobots.io/kickstarter-datasets/)

2. Data Merger
- Merges the clean dataset

3. Main Code (code_with_ml.py)
- Creates the key variables and runs the regression discontinuity
- Provides a LGBM regressor to improve identification and provide weights for regression discontinuity
- Robustness checks
  1. Bandwidth Robustness
  2. Threshold Robustness
  3. Donut Hole RD Estimates
  4. Bandwidth by Gender Estimates
- Plots for regression discontinuity

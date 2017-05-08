# Recalibration
FIDUCEO modules for re-calibration of a satellite sensor series to a reference sensor

Scripts in this repo perform calibration of a satellite sensor to a reference, tested against AVHRR simulated data. The AVHRR series is implemented as a class that includes the measurement equation, its partial derivatives, calibration uncertainty evaluated by GUM law of propagation. The (re-)calibration is performed as an error-in-variables (EIV) regression using ODRPACK implementation in SciPy on matchup data of a pair series sensor with reference sensor. 

Two main scripts call the other modules, the script MCerrst.py and MCrnd.py. 

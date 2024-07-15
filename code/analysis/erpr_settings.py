# String indicating path to edf2asc.exe file;
# This executable converts eyelink data files to ascii files.
# Only the latter type can be read in python.
# Install the eyelink developers kit  (EDK) to get edf2asc.exe for command.
# DO NOT USE the edfConverter.exe program
# The EDK can be found on www.SR-support.com after signing up on the forum.
# Getting access to the SR-research forum might take some time until approved.
# NOT TESTED BUT YOU MIGHT HAVE TO ADD edf2asc.exe to the PATH variable of user's specific environment variables

edf2ascPath = "C:/Program Files (x86)/SR Research/EyeLink/bin/"

# boolean for showing full pupil traces
showPupilTraces = False

# boolean for skipping already analyzed files
skipAlrAnFiles = True

# boolean for interpolating missing blink periods
interpBlinks = True

# integer for the maximum duration of blink periods to be interpolated in milliseconds
maxInterpBlinkPeriod = 300

# integer for range of baseline period calculations in milliseconds
pBasePer = [0, 250]

# integers for range of pupil response calculations (mostly a constriction)
pERPROriAmpPer = [250, 1750]
# pERPROriAmpPer = [250, 1250] # worse results
pERPROriAreaPer = [250, 2000]

# integers for range of pupil post-response (e.g., alerting/effort related to button press response)
pPostERPRPer = 1900

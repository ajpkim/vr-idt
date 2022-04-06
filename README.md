# vr-idt

This package provides a VR tailored eye-tracking classification algorithm for identifying fixations and saccades in data gathered from VR headsets.

## Background

The algorithm was initially proposed and implemented by Jose Llanes-Jurado, Javier Marín-Morales, Jaime Guixeres, and Mariano Alcañiz in their paper, [Development and Calibration of an Eye-Tracking Fixation Identification Algorithm for Immersive Virtual Reality](https://www.mdpi.com/1424-8220/20/17/4956).

Code from the original authors is here [github/ASAPLableni](https://github.com/ASAPLableni/VR-centred_I-DT_algorithm).

## Installation

Install this package using `pip` ([Python virtual environments](https://docs.python.org/3/library/venv.html)).

`$ pip install vr-idt`

## Documentation

```python
def classify_fixations(df: pd.DataFrame,
                       min_duration: float = 0.15,
                       max_angle: float = 1.50,
                       min_freq: float = 30.0,
                       time: str = "time",
                       gaze_world_x: str = "gaze_world_x",
                       gaze_world_y: str = "gaze_world_y",
                       gaze_world_z: str = "gaze_world_z",
                       head_pos_x: str = "head_pos_x",
                       head_pos_y: str = "head_pos_y",
                       head_pos_z: str = "head_pos_z") -> pd.DataFrame:
    """Classify VR eye fixation events in eye-tracking data.

    Args:
        df: DataFrame with eye tracking data to classify
        min_duration: The minimum length of a fixation in seconds
        max_angle: The maximum angle of dispersion within the fixation within
        min_freq: The minimum required frequency for a fixation to be classified
        time: df column name for time (sec) data
        gaze_world_x: df column name for gaze position in virtual world data
        gaze_world_y: df column name for gaze position in virtual world data
        gaze_world_z: df column name for gaze position in virtual world data
        head_pos_x: df column name for head position in physical space data
        head_pos_y: df column name for head position in physical space data
        head_pos_z: df column name for head position in physical space data
    Returns:
        fixation_df: Copy of original arg 'df' with 4 new fixation related columns:
            "fixation", "fixation_start", "fixation_end", and "fixation_duration"
    """
```

## Example

```python
import pandas as pd

from vr_idt.vr_idt import classify_fixations

# Load in data with eye tracking data
df = pd.read_csv("<path/to/data>")

# Setup a column name mapping so algorithm knows where to look for necessary data
col_name_map = {
	"gaze_world_x": "Gaze Pos X (world)",
	"gaze_world_y": "Gaze Pos Y (world)",
	"gaze_world_z": "Gaze Pos Z (world)",
	"head_pos_x": "Head Pos X",
	"head_pos_y": "Head Pos Y",
	"head_pos_z": "Head Pos Z"
	}

# Define some parameters
min_duraion = 0.15
max_angle = 1.5
min_freq = 25

# Run algorithm and add 4 fixation related columns to df
df = classify_fixations(df, min_duration, max_angle, min_freq, **col_name_map)
```

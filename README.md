# Depth and Instance Segmentation-Based Speed Estimation
This project demonstrates how to calculate the speed of objects using depth estimation and instance segmentation.

## Steps to Run the Project
### Prerequisites
- Ensure Python 3.8+ is installed.
- Install necessary dependencies listed in the repositories mentioned below.

### Setup Instructions
1. Clone the following repositories:
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2
git clone https://github.com/DepthAnything/Depth-Anything-V2
```
2. Install dependencies for both repositories

3. Place the input video car3.mp4 in the following directories:

- For depth estimation: Depth-Anything-V2/metric_depth/
- For instance segmentation: Grounded-SAM-2/

### Running the Scripts
1. Run Depth Estimation
Execute run_video.py from Depth-Anything-V2:

```bash
python run_video.py --video-path metric_depth/car3.mp4 --outdir metric_depth/vis_depth_car3 --save-numpy
```
2. Run Instance Segmentation
Execute car_segmentation.py from Grounded-SAM-2:

```bash
python car_segmentation.py
```
This script will generate segmentation masks and annotations for the input video.

3. Compute Object Velocity
After completing both scripts above, execute compute_velocity.py:

```bash
python compute_velocity.py
```
This script calculates the speed of objects based on the depth and instance segmentation results.

### Output
The computed speeds and visualized frames with overlaid information will be saved as an output video in Grounded-SAM-2/.
### Notes
Ensure proper paths are set for car3.mp4 and output directories in the scripts.
Adjust parameters like FPS, resolution, or thresholds in the scripts if needed.
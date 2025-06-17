## QUELLE: https://github.com/megapose6d/megapose6d/issues/52

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Load the image
image_path = "megapose6d/local_data/examples/barbecue-sauce/image_rgb.png"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image from {image_path}")
    exit(1)

# Extract camera parameters
K = np.array([
    [605.9547119140625, 0.0, 319.029052734375],
    [0.0, 605.006591796875, 249.67617797851562],
    [0.0, 0.0, 1.0]
])

# Extract pose parameters
label = "barbecue-sauce"
quaternion = np.array([00.5439271089053248, 0.6215568366567065, -0.434985966847834, 0.35860516257043684])
translation = np.array([0.1073746383190155, 0.07319149374961853, 0.4580971896648407])

# Convert quaternion to rotation matrix
rot = Rotation.from_quat(quaternion)
R = rot.as_matrix()

# Create transformation matrix
transform = np.eye(4)
transform[:3, :3] = R
transform[:3, 3] = translation

# Define a 3D bounding box for purion (approximate dimensions in meters)
# Adjust these dimensions to match the actual object size
half_width, half_depth, half_height = 0.06, 0.045, 0.09
box_3d = np.array([
    [-half_width, -half_depth, -half_height],  # Bottom face
    [half_width, -half_depth, -half_height],
    [half_width, half_depth, -half_height],
    [-half_width, half_depth, -half_height],
    [-half_width, -half_depth, half_height],   # Top face
    [half_width, -half_depth, half_height],
    [half_width, half_depth, half_height],
    [-half_width, half_depth, half_height]
])

# Transform the 3D box vertices to world coordinates
box_transformed = []
for point in box_3d:
    # Create homogeneous coordinate
    point_homog = np.append(point, 1)
    # Apply transformation
    transformed_point = transform @ point_homog
    # Convert back to 3D point
    box_transformed.append(transformed_point[:3])

# Project the 3D points onto the 2D image plane
box_2d = []
for point_3d in box_transformed:
    # Project using the camera intrinsics
    point_2d = K @ point_3d
    point_2d = point_2d / point_2d[2]  # Normalize by Z
    box_2d.append(point_2d[:2])

box_2d = np.array(box_2d, dtype=np.int32)

# Define the connections between box vertices
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
]

# Draw the projected box on the image
for edge in edges:
    start_point = (box_2d[edge[0]][0], box_2d[edge[0]][1])
    end_point = (box_2d[edge[1]][0], box_2d[edge[1]][1])
    cv2.line(img, start_point, end_point, (0, 255, 0), 2)

# Add coordinate axes to visualize orientation (X=red, Y=green, Z=blue)
origin = transform[:3, 3]
axes_length = 0.05  # 5cm axes

for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # BGR: Z=blue, Y=green, X=red
    axis_end = origin + axes_length * transform[:3, i]
    
    # Project origin and axis endpoint to image
    origin_2d = K @ origin
    origin_2d = (int(origin_2d[0] / origin_2d[2]), int(origin_2d[1] / origin_2d[2]))
    
    axis_2d = K @ axis_end
    axis_2d = (int(axis_2d[0] / axis_2d[2]), int(axis_2d[1] / axis_2d[2]))
    
    # Draw axis line
    cv2.line(img, origin_2d, axis_2d, color, 2)

# Draw 2D bounding box from bbox_modal if available
# Uncomment and adjust if you have 2D bbox information
bbox_modal = [1.61, 295.08, 622.31, 475.98]
if bbox_modal:
    xmin, ymin, xmax, ymax = [int(coord) for coord in bbox_modal]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue rectangle for 2D bbox

# Add label
cv2.putText(img, label, (box_2d[0][0], box_2d[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image
output_path = "megapose6d/local_data/examples/barbecue-sauce/test.png"
cv2.imwrite(output_path, img)
print(f"Visualization saved to {output_path}")

# Display the image with proper exit handling
print("Press any key to close the visualization window")
cv2.imshow('Object Pose Visualization', img)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
cv2.waitKey(1)  # This additional waitKey call ensures window closes properly on some systems

# Exit program
print("Visualization complete and window closed")
"""
.. _images_with_landmarks:

Display 3D Images and Landmarks with Interactive Sliders
=========================================================

This example demonstrates how to visualize 3D+t medical images using interactive Matplotlib sliders.
It also shows how to overlay landmarks and compose multiple visualization modules that share the same
figure and navigation controls (sliders).

The viewer is modular: any subclass of `Base3dAxes_slider` can reuse the same figure and sliders to
enable synchronized display of multiple data layers (e.g., images, landmarks, vector fields).

Three use cases are presented:
    1. Visualizing a single image volume
    2. Adding landmark overlays in shared context
    3. Using a helper function to compare two images and their landmarks
"""

#############################################
# import the necessary packages
import os
import torch
import demeter.utils.axes3dsliders_plt as a3s
#%%
#############################################
 # Load a 3D+t volume and visualize it
# ------------------------------------
# This block loads a longitudinal dataset from a `.pt` file and displays one volume
# using the `Image3dAxes_slider` viewer.
# You can choose between FLAIR, T1CE, or other outputs (like prediction maps).


path = "/home/turtlefox/Documents/11_metamorphoses/data/pixyl/aligned/PSL_001/"
file = "PSL_001_longitudinal_rigid.pt"
data = torch.load(os.path.join(path, file))

months = data["months_list"]
flair = data["flair_longitudinal"]
t1ce = data["t1ce_longitudinal"]
pred = data["pred_longitudinal"]

# Normalize images for display
flair /= flair.max()
t1ce /= t1ce.max()

# Choose one modality to visualize
img = flair
# img = tb.temporal_img_cmp(flair, t1ce)
# img = torch.clip(pred, 0, 10)
# img = flair[-1,0]

ias = a3s.Image3dAxes_slider(img)
# ias.change_image(pred, cmap='tab10')

# Optionally move to a specific slice
ias.go_on_slice(168,176,53)
ias.show()

#############################################
# Add landmark overlays (shared sliders)
# --------------------------------------
# This block shows how to overlay landmarks on the same figure as the image.
# All viewers share the same context (`ctx`) for synchronized navigation.


#%%
_,_,H,W,D  = img.shape
## random landmarks
# lh = torch.randint(0,H,(10,1))
# lw = torch.randint(0,W,(10,1))
# ld = torch.randint(0,D,(10,1))
## hand crafted landmarks
ld = torch.tensor([193, 200, 158, 178, 205, 212])[:,None]
lh = torch.tensor([231,160, 151, 269, 225, 93])[:,None]
lw = torch.tensor([50, 50, 54, 65, 85, 47])[:,None]



landmarks = torch.cat((ld, lh, lw), dim=1)
print('\n Iandmark shape: ', landmarks.shape)
print(landmarks)
#%%
# Create the main image viewer
ias = a3s.Image3dAxes_slider(img)

 # Add landmark viewer using the shared context from ias
las = a3s.Landmark3dAxes_slider(landmarks,
                            image_shape = (1,H,W,D,1),
                            shared_context=ias.ctx
                            )
las.show()


##############################################
# Use the helper function to compare two images + landmarks
# ---------------------------------------------------------
# This example demonstrates the `compare_images_with_landmarks` helper which
# automatically creates a viewer with:
#     - two images (image0 and image1)
#     - landmark overlays
#     - buttons to switch between views (image0, image1, and composed comparison)

# Generate synthetic variation on landmarks for demo purposes

noise = torch.randint(-3,3,landmarks.shape)
a3s.compare_images_with_landmarks(
    image0=flair[-1][None],  # Final FLAIR image
    image1=t1ce[-1][None],
    landmarks0=landmarks,
    landmarks1=landmarks + noise,
    method="compose"  # Use RGB composition to compare image0 and image1
)
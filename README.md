# MirrorWorld

Project Overview


​This script generates a 3D model of a specified real-world location. It fetches building and road data from OpenStreetMap and elevation data from the OpenTopography API. The script then constructs and renders a textured 3D scene using PyVista, offering both an interactive view and the ability to save a screenshot from a specific, camera-matched perspective. The system includes a local caching mechanism to speed up subsequent runs for the same location.
​Workflow for High-Fidelity Texture Mapping
​To apply a texture from a real photograph onto a 3D building model, you must mathematically link the 2D photo to the 3D world space. This workflow describes a semi-automated process to achieve this.
​Step 1: Prepare 3D-2D Correspondence Points
​This is the most critical manual step. You must identify the same points (e.g., building corners) in both your 3D model and your 2D reference photograph.
​Get 3D World Coordinates: Temporarily modify the populate_plotter function to print the UTM coordinates of the building you want to texture. Identify the target building by its osmid. You need at least four corner points that form a plane (e.g., a facade).
​Get 2D Pixel Coordinates: Open your reference photograph in any image editor (like GIMP, Photoshop, or even MS Paint). Hover your mouse over the exact same four corner points and record their (x, y) pixel coordinates.
​You now have a set of 3D points and their corresponding 2D pixel locations in the photo.
​Step 2: Extract and Warp Texture with OpenCV
​With the correspondence points, you can now use OpenCV to perform a perspective transform. This will "un-distort" the building facade from the photograph into a flat, rectangular texture file.
​The script below performs this operation. It reads the source image, applies the transformation based on your four 2D points, and saves a clean texture file.
​Step 3: Apply Custom Texture in Main Script
​Save the new texture (e.g., warped_texture.jpg) into the textures directory. The main create_3d_model.py script has been updated to handle this. You can specify a target osmid and the custom texture you want to apply to it.

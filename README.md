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



## Future Development & Research Directions

This project has a strong foundation. To push it further towards photorealism, you can explore these advanced topics.

1. Automated Camera Pose Estimation

Instead of manually aligning the camera, you can automate it. The cv2.solvePnP function in OpenCV is designed for this. It takes the 3D world points and 2D image points you prepared and calculates the precise camera rotation and translation vectors. This gives you a mathematically perfect alignment.

2. Neural Radiance Fields (NeRF)

NeRF is a state-of-the-art technique that learns a continuous 3D representation of a scene from a collection of 2D images.

    How it works: A neural network maps a 5D coordinate (3D location x,y,z and 2D viewing direction θ,φ) to a single volume density and color. By rendering rays through this learned field, it can generate photorealistic novel views.

    Pros: Incredible realism for view synthesis.

    Cons: It doesn't produce an explicit mesh like this project does, making it harder to edit or use in traditional game engines. Training also requires many photos from different angles.

3. Inverse Rendering and "Delighting"

A major challenge is that a real photograph has lighting and shadows "baked into" the pixels. When you apply this as a texture, it clashes with the 3D engine's own lighting.

    Inverse Rendering: A field of computer graphics that aims to decompose an image into its intrinsic properties: albedo (the pure, lightless color of a surface), specularity, roughness, and normals.

    Delighting: A sub-problem focused on removing shadows and lighting information from an image to recover the albedo map. This is extremely challenging but is key to creating reusable, high-quality textures from single photos. AI-based techniques are the current state-of-the-art here.

4. Professional Photogrammetry Software

For large-scale projects, professionals use dedicated software that automates this entire pipeline.

    Examples: Agisoft Metashape, RealityCapture, COLMAP (open-source).

    Process: You feed them dozens or hundreds of photos of an object or area. They automatically perform feature matching, calculate all camera positions, and generate a dense, textured 3D mesh. This is the industry-standard approach for creating "digital twins."
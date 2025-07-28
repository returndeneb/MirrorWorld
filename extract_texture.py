import cv2
import numpy as np
import argparse

def extract_and_warp_texture(image_path: str, source_points_2d: list, output_width: int, output_height: int, output_filename: str):
    """
    Extracts a region from an image and warps it into a rectangular texture.

    Args:
        image_path: Path to the source image file.
        source_points_2d: A list of four [x, y] pixel coordinates from the source image.
                          The order should be top-left, top-right, bottom-right, bottom-left.
        output_width: The desired width of the output texture in pixels.
        output_height: The desired height of the output texture in pixels.
        output_filename: The filename for the saved texture.
    """
    # 1. Load the source image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Define source and destination points for the perspective transform
    src_pts = np.array(source_points_2d, dtype=np.float32)

    # Destination points are the corners of the new output image
    dst_pts = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    # 3. Calculate the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 4. Apply the perspective warp
    warped_texture = cv2.warpPerspective(image, transform_matrix, (output_width, output_height))

    # 5. Save the resulting texture
    cv2.imwrite(output_filename, warped_texture)
    print(f"Successfully extracted texture to '{output_filename}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract and warp a texture from a source image.")
    parser.add_argument("--source_image", type=str, required=True, help="Path to the source photograph.")
    parser.add_argument("--points", type=int, nargs=8, required=True, metavar=('TL_X', 'TL_Y', 'TR_X', 'TR_Y', 'BR_X', 'BR_Y', 'BL_X', 'BL_Y'),
                        help="Pixel coordinates of the four corners: top-left, top-right, bottom-right, bottom-left.")
    parser.add_argument("--width", type=int, default=512, help="Width of the output texture file.")
    parser.add_argument("--height", type=int, default=512, help="Height of the output texture file.")
    parser.add_argument("--output_file", type=str, default="custom_texture.png", help="Filename for the output texture.")
    args = parser.parse_args()

    # Reshape the points argument into a list of lists
    points_list = [
        [args.points[0], args.points[1]],
        [args.points[2], args.points[3]],
        [args.points[4], args.points[5]],
        [args.points[6], args.points[7]],
    ]
    
    extract_and_warp_texture(args.source_image, points_list, args.width, args.height, args.output_file)

# Example command:
# python extract_texture.py --source_image "real_photo.jpg" --points 1150 450 1400 465 1410 750 1160 720 --width 512 --height 1024 --output_file "building_facade.png"
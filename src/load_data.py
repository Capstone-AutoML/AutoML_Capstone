# Load images from local storage
import os


def load_images(image_dir: str) -> list:
    """
    Load images from a directory and return a list of image file paths.

    Parameters
    ----------
    image_dir: str
        The directory containing the images.

    Returns
    -------
    list
        A list of image file paths.
    """
    image_paths = []
    for file_name in os.listdir(image_dir):
        ext = os.path.splitext(file_name)[-1].lower()
        if ext in {'.jpg', '.jpeg', '.png'}:
            image_paths.append(os.path.join(image_dir, file_name))
    return image_paths

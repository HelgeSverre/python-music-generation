import os


def get_unique_filename(base_filename):
    """
    Generate a unique filename by appending a number if the file already exists.
    """
    if not os.path.exists(base_filename):
        return base_filename

    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

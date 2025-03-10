import os
import platform
import shutil
import sys

def install_font(font_path):
    # Check if font file exists
    if not os.path.exists(font_path):
        print(f"Font file {font_path} does not exist.")
        return False

    # Determine the operating system
    current_os = platform.system().lower()

    if current_os == 'windows':
        return install_font_windows(font_path)
    elif current_os == 'darwin':
        return install_font_mac(font_path)
    elif current_os == 'linux':
        return install_font_linux(font_path)
    else:
        print(f"Unsupported operating system: {current_os}")
        return False


def install_font_windows(font_path):
    font_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
    try:
        font_name = os.path.basename(font_path)
        dest_path = os.path.join(font_dir, font_name)
        
        # Copy the font file to the Windows Fonts folder
        shutil.copy(font_path, dest_path)
        print(f"Font installed successfully at {dest_path}")
        return True
    except Exception as e:
        print(f"Error installing font on Windows: {e}")
        return False


def install_font_mac(font_path):
    user_font_dir = os.path.expanduser('~/Library/Fonts')
    system_font_dir = '/Library/Fonts'
    
    # First try installing for the current user
    try:
        dest_path = os.path.join(user_font_dir, os.path.basename(font_path))
        shutil.copy(font_path, dest_path)
        print(f"Font installed successfully at {dest_path}")
        return True
    except Exception as e:
        print(f"Error installing font for user on macOS: {e}")

    # If the user folder fails, try the system-wide directory (requires admin access)
    try:
        dest_path = os.path.join(system_font_dir, os.path.basename(font_path))
        shutil.copy(font_path, dest_path)
        print(f"Font installed system-wide at {dest_path}")
        return True
    except Exception as e:
        print(f"Error installing font system-wide on macOS: {e}")
        return False


def install_font_linux(font_path):
    user_font_dir = os.path.expanduser('~/.fonts')
    system_font_dir = '/usr/share/fonts'

    try:
        # Create the user's font directory if it doesn't exist
        if not os.path.exists(user_font_dir):
            os.makedirs(user_font_dir)

        dest_path = os.path.join(user_font_dir, os.path.basename(font_path))
        shutil.copy(font_path, dest_path)
        print(f"Font installed successfully at {dest_path}")
        return True
    except Exception as e:
        print(f"Error installing font for user on Linux: {e}")

    # If the user folder fails, try the system-wide directory (requires sudo)
    try:
        dest_path = os.path.join(system_font_dir, os.path.basename(font_path))
        shutil.copy(font_path, dest_path)
        print(f"Font installed system-wide at {dest_path}")
        return True
    except Exception as e:
        print(f"Error installing font system-wide on Linux: {e}")
        return False


if __name__ == '__main__':
    # Example font path (update to your .ttf file path)
    font_path = os.path.join('fonts', 'Kranky.ttf')
    
    if install_font(font_path):
        print("Font installation successful.")
    else:
        print("Font installation failed.")

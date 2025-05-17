###########################################
# nii_to_pngs_converter.py for Python 3   #
#         NIfTI Image Converter           #
#                                         #                       
#     Written by Monjoy Saha              #
#        monjoybme@gmail.com              #
#          02 July 2020                   #
#                                         #
###########################################


import numpy, os, nibabel
from matplotlib import image

def convert_nii_to_png(file_path, output_dir_path=None, rotation_angle=90):
    """
    Convert a single NIfTI file to PNG images.
    
    Args:
        file_path (str): Path to the NIfTI file (.nii or .nii.gz)
        output_dir_path (str, optional): Directory where PNG files should be saved. 
                                       If None, creates a directory next to the input file.
        rotation_angle (int): Rotation degree (90°, 180°, 270°). Default is 90°
        
    Returns:
        str: Path to the output directory containing the PNG files
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the input file is not a NIfTI file
        
    Example:
        >>> output_dir = convert_nii_to_png("brain_scan.nii", output_dir_path="output/pngs")
        >>> print(f"PNG files saved to: {output_dir}")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    if not file_path.endswith(('.nii', '.nii.gz')):
        raise ValueError(f"Input file must be a NIfTI file (.nii or .nii.gz): {file_path}")

    fname = os.path.basename(file_path)
    fname_noext = fname.split('.', 1)[0]
    
    # Determine output directory
    if output_dir_path is None:
        # Default behavior: create directory next to input file
        base_path = os.path.dirname(os.path.abspath(file_path))
        output_dir = os.path.join(base_path, fname_noext)
    else:
        # Use provided output directory path
        output_dir_path = os.path.abspath(output_dir_path)
        output_dir = os.path.join(output_dir_path, fname_noext)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load and get image data
    image_nib = nibabel.load(file_path)
    image_array = image_nib.get_fdata()
    print(f"Image dimensions: {len(image_array.shape)}D")
    
    slice_counter = 0
    
    # For 3D image input    
    if len(image_array.shape) == 3:
        nx, ny, nz = image_array.shape
        total_slices = nz
        # iterate through slices
        for current_slice in range(0, total_slices):
            # rotate or no rotate
            if rotation_angle == 90:
                data = numpy.rot90(image_array[:, :, current_slice])
            elif rotation_angle == 180:
                data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice]))
            elif rotation_angle == 270:
                data = numpy.rot90(numpy.rot90(numpy.rot90(image_array[:, :, current_slice])))
            
            print(f'Saving slice {current_slice + 1}/{total_slices}...')
            image_name = f"{fname_noext}_z{str(current_slice+1).zfill(3)}.png"
            image_name = os.path.join(output_dir, image_name)
            image.imsave(image_name, data, cmap='gray')
            slice_counter += 1
        print('Finished converting 3D image')
                
    elif len(image_array.shape) == 4:
        nx, ny, nz, nw = image_array.shape
        total_volumes = nw
        total_slices = nz
        for current_volume in range(0, total_volumes):
            # iterate through slices
            for current_slice in range(0, total_slices):
                if rotation_angle == 90:
                    data = numpy.rot90(image_array[:, :, current_slice, current_volume])
                elif rotation_angle == 180:
                    data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice, current_volume]))
                elif rotation_angle == 270:
                    data = numpy.rot90(numpy.rot90(numpy.rot90(image_array[:, :, current_slice, current_volume])))

                print(f'Saving volume {current_volume + 1}/{total_volumes}, slice {current_slice + 1}/{total_slices}...')
                image_name = f"{fname_noext}_vol{current_volume+1}_z{str(current_slice+1).zfill(3)}.png"
                image_name = os.path.join(output_dir, image_name)
                image.imsave(image_name, data, cmap='gray')
                slice_counter += 1
            print(f'Finished converting volume {current_volume + 1}/{total_volumes}')
    
    print(f'Successfully converted {slice_counter} slices to PNG')
    return output_dir

# CLI functionality
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert a NIfTI file to PNG images')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input NIfTI file (.nii or .nii.gz)')
    parser.add_argument('--output_dir_path', type=str, default=None,
                        help='Directory where PNG files should be saved (optional)')
    parser.add_argument('--rotation_angle', type=int, default=90, 
                        help='Rotation degree, i.e., 90°, 180°, 270°, default value is 90°')
    args = parser.parse_args()
    
    try:
        output_dir = convert_nii_to_png(args.input_file, args.output_dir_path, args.rotation_angle)
        print(f"PNG files saved to: {output_dir}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")
        exit(1)


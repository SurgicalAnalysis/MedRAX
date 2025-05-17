from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import uuid
import tempfile
import numpy as np
import nibabel as nib
from PIL import Image
from pydantic import BaseModel, Field
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class NiftiProcessorInput(BaseModel):
    """Input schema for the NIfTI Processor Tool."""

    nifti_path: str = Field(..., description="Path to the NIfTI file (.nii or .nii.gz)")
    rotation_angle: Optional[int] = Field(
        90, description="Rotation angle for the output image (90, 180, or 270 degrees)"
    )


class NiftiProcessorTool(BaseTool):
    """Tool for processing NIfTI files and converting them to PNG images."""

    name: str = "nifti_processor"
    description: str = (
        "Processes NIfTI medical image files and converts them to standard image format. "
        "Handles 3D and 4D NIfTI files, extracting 2D slices. "
        "Currently configured to return only the first slice, but can be modified for multiple slices. "
        "Input: Path to NIfTI file (.nii or .nii.gz) and optional rotation angle. "
        "Output: Path to processed image file and NIfTI metadata."
    )
    args_schema: Type[BaseModel] = NiftiProcessorInput
    temp_dir: Path = None

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize the NIfTI processor tool."""
        super().__init__()
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    def _rotate_image(self, img_array: np.ndarray, rotation_angle: int) -> np.ndarray:
        """Apply rotation to the image array."""
        rotations = rotation_angle // 90
        return np.rot90(img_array, k=rotations)

    def _process_nifti(
        self,
        nifti_path: str,
        rotation_angle: int = 90,
    ) -> Tuple[np.ndarray, Dict]:
        """Process NIfTI file and extract metadata."""
        # Load NIfTI file
        nii_img = nib.load(nifti_path)
        img_array = nii_img.get_fdata()
        
        # Extract metadata
        metadata = {
            "dimensions": len(img_array.shape),
            "shape": img_array.shape,
            "affine": nii_img.affine.tolist(),
            "header_info": dict(nii_img.header),
        }

        # MODIFICATION POINT #1:
        # Currently, we only process the first slice. To process multiple slices,
        # you can modify this section to return multiple slices or implement a
        # slice selection parameter.
        
        if len(img_array.shape) == 4:
            # For 4D images, take first volume and first slice
            slice_data = img_array[:, :, 0, 0]
        else:  # 3D image
            # Take first slice
            slice_data = img_array[:, :, 0]

        # MODIFICATION POINT #2:
        # To process multiple slices, you could:
        # 1. Add a slice_range parameter to NiftiProcessorInput
        # 2. Create a loop here to process multiple slices
        # 3. Return a list of processed images instead of a single image
        # Example modification:
        """
        processed_slices = []
        for slice_idx in range(slice_range[0], slice_range[1]):
            if len(img_array.shape) == 4:
                slice_data = img_array[:, :, slice_idx, 0]
            else:
                slice_data = img_array[:, :, slice_idx]
            processed_slices.append(self._rotate_image(slice_data, rotation_angle))
        """

        # Rotate the slice
        processed_slice = self._rotate_image(slice_data, rotation_angle)
        
        # Normalize to 0-255 range
        processed_slice = ((processed_slice - processed_slice.min()) / 
                         (processed_slice.max() - processed_slice.min()) * 255).astype(np.uint8)

        return processed_slice, metadata

    def _run(
        self,
        nifti_path: str,
        rotation_angle: int = 90,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Process NIfTI file and save as viewable image.

        Args:
            nifti_path: Path to input NIfTI file
            rotation_angle: Rotation angle in degrees (90, 180, or 270)
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict, Dict]: Output dictionary with processed image path and metadata dictionary
        """
        try:
            # Process NIfTI and save as PNG
            img_array, metadata = self._process_nifti(nifti_path, rotation_angle)
            output_path = self.temp_dir / f"processed_nifti_{uuid.uuid4().hex[:8]}.png"
            Image.fromarray(img_array).save(output_path)

            output = {
                "image_path": str(output_path),
            }

            metadata.update({
                "original_path": nifti_path,
                "output_path": str(output_path),
                "rotation_angle": rotation_angle,
                "analysis_status": "completed",
            })

            return output, metadata

        except Exception as e:
            return (
                {"error": str(e)},
                {
                    "nifti_path": nifti_path,
                    "analysis_status": "failed",
                    "error_details": str(e),
                },
            )

    async def _arun(
        self,
        nifti_path: str,
        rotation_angle: int = 90,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Async version of _run."""
        return self._run(nifti_path, rotation_angle)
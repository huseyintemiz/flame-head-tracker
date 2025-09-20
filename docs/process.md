Key Steps of Face Alignment:

  1. Landmark Extraction (lines 152-161)

  The function takes 68 facial landmarks and groups them
  into:
  - Chin (0-17)
  - Eyebrows (17-27)
  - Nose (27-36)
  - Eyes (36-48)
  - Mouth (48-68)

  2. Calculate Reference Points (lines 164-171)

  - Computes eye centers by averaging left/right eye
  landmarks
  - Calculates eye-to-eye vector (horizontal alignment
  reference)
  - Finds mouth center from outer mouth corners
  - Computes eye-to-mouth vector (vertical alignment
  reference)

  3. Determine Crop Rectangle (lines 175-182)

  - Creates an oriented coordinate system using eye-to-eye
  and eye-to-mouth vectors
  - Scales the crop size based on face dimensions (2.0x eye
   distance or 1.8x mouth-to-eye distance)
  - Centers the crop slightly below the eyes (10% toward
  mouth)
  - For 'tracking' mode, applies 1.3x scale factor for
  wider crop

  4. Image Processing:

  - Shrinking (lines 195-200): Downsamples if face is too
  large
  - Cropping (lines 203-208): Crops to face region with 10%
   border
  - Padding (lines 211-226): Adds padding if face is near
  image edge, with optional reflection or constant padding
  - Transform (lines 229-231): Applies quadrilateral
  transformation to align face to standard orientation

  5. Final Output

  Returns aligned face image at specified resolution
  (default 256x256), with face:
  - Upright (eyes horizontal)
  - Centered
  - Scaled consistently
  - Optionally padded for edge cases

  The alignment ensures consistent face orientation and
  scale across different input images, crucial for FLAME
  tracking and reconstruction tasks.
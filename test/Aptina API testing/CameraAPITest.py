import sys
from win32com.client import Dispatch
import cv2
import numpy as np

def main():
    try:
        # Create the ApBase COM object
        apbase = Dispatch('apbaseCom.ApBase')

        # Probe the system and create the first camera device found.
        # If no probe done yet, Create(0) will do a DeviceProbe() internally.
        camera = apbase.Create(0)

        # Initialize the camera with default settings
        # Passing empty strings uses the default init preset.
        ini = r'D:\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Budgies\Pupillometry\PupilCam\Kerr Lab Eye Camera\[Docs] Dual-eye cam\MT9V024-REV4.ini';
        err_code = camera.LoadIniPreset(ini, 'EyeCAM')
        if err_code != 256:
            print(f"Initialization returned error code {err_code}")
            return

        width = camera.Width
        height = camera.Height
        imageType = camera.ImageType
        print('Camera outputs {w}x{h} {t}'.format(w=width, h=height, t=imageType))

        # Discard the first frame (as recommended by documentation)
        _ = camera.GrabFrame()

        while True:
            # Grab a new frame
            raw_frame = camera.GrabFrame()
            print('last error:', apbase.LastError)
            if apbase.LastError != 0:
                print(f"Error grabbing frame: {apbase.LastError}")
                return

            # Convert the raw frame to RGB
            # Note: ColorPipe returns a 1D array. You can reshape it later.
            rgb_frame = camera.ColorPipe(raw_frame)
            print('last error:', apbase.LastError)
            if apbase.LastError != 0:
                print(f"Error converting frame: {apbase.LastError}")
                return

            print("Frame grabbed and converted successfully!")
            print(rgb_frame)
            print(type(rgb_frame))
            # rgb_frame is now an array of B, G, R, X values for each pixel.
            # You can reshape and display/save it using your preferred image processing library.

            # Convert the returned VARIANT array to a numpy array
            rgb_data = np.array(rgb_frame, dtype=np.uint8)

            # rgb_data is in BGRA (B, G, R, X) format
            # Reshape to (height, width, 4)
            rgb_data = rgb_data.reshape((height, width, 4))

            # Extract just the BGR channels
            bgr_data = rgb_data[:, :, :3]

            # Display the image
            cv2.imshow('Live View', bgr_data)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()

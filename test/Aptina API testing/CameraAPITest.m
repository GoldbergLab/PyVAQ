% open ApBase and Camera, and initialize via "Demo Initialization" preset
apbase = actxserver('apbaseCom.ApBase');
camera = apbase.Create(0);
ini = 'D:\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Budgies\Pupillometry\PupilCam\Kerr Lab Eye Camera\[Docs] Dual-eye cam\MT9V024-REV4.ini';
camera.LoadIniPreset(ini, 'EyeCAM');
camera.Index
camera.Name
camera.PartNumber
camera.Version
camera.VersionName
camera.FileName
camera.Width
camera.Height
camera.ImageType
camera.ShipAddr
camera.CameraSerialNumber

% Throw-away the first frame after initialization
raw = camera.GrabFrame;

% grab a frame, convert to RGB
while true
    raw = camera.GrabFrame;
    rgb = camera.ColorPipe(raw);
    
    rgbz = double(rgb);
    rgbz = (rgbz - min(rgbz)) / (max(rgbz) - min(rgbz));
    imshow(reshape(rgbz, camera.width, camera.height)');
end
So you're trying to install PyVAQ. Here's how:

1. Install GitHub Desktop
2. Clone PyVAQ repository to your computer (https://github.com/GoldbergLab/PyVAQ)
3. Install NI-DAQmx (driver for DAQ)
4. Install Python 3.8.10
5. Install python libraries:
	a) Open command prompt
	b) cd C:\path\to\where\PyVAQ\is
	c) pip install -r requirements.txt
6. Install Spinnaker SDK
7. Install Spinnaker python library:
	a) Unzip spinnaker-python
	b) Open command prompt
	c) cd C:\path\to\where\unzipped\spinnaker-python\is
	d) pip install spinnaker_python-2.4.0.144-cp38-cp38-win_amd64.whl 
8. Install ffmpeg with GPU support
	a) Unzip ffmpeg
	b) Move to C:\ProgramFiles
	c) Add ffmpeg path to system Path (environment variable)
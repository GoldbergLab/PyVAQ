import sys
from pathlib import Path
import wave
import shutil
import subprocess

FFMPEG_EXE = shutil.which('ffmpeg')

# A script to split mutli-channel audio files into separate single-channel audio
#   files, plus one file that has both stereo tracks mixed into one mono track

def splitAudioTracksInFolder(folderPaths, overwrite=False, dryRun=False, requireNumericalEndTag=True):
    '''Loop through files in a folder and split audio files into track files

    Arguments:
        folderPaths = one or more paths to the folder to look in (either a
            string or a pathlib.Path object)
        overwrite = (optional) boolean flag indicating whether or not to
            overwrite files if they already exist. Default is False.
        dryRun = (optional) boolean flag indicating whether or not to do a
            dry run, printing the expected behavior rather than doing it.
            Default is False.
        requireNumericalEndTag = (optional) optional boolean flag indicating that
            a numerical end-tag to the filename is required. This will restrict
            the action of the script to audio files with the filename format
            produced by PyVAQ running SimpleAudioWriter.
    '''

    if type(folderPaths) != type([]):
        # User passed a single path in - wrap it in a list for consistency.
        folderPaths = [folderPaths]

    audioFiles = []
    for folderPath in folderPaths:
        for k, subpath in enumerate(folderPath.iterdir()):
            if subpath.is_file():
                extension = subpath.suffix.lower()
                if extension == '.wav':
                    audioFiles.append(subpath)

    fileCount = 0
    splitCount = 0

    for audioFile in audioFiles:
        extension = audioFile.suffix.lower()
        name = audioFile.name[:-len(extension)]

        # Check that file matches expected filename pattern
        if '_' not in name:
            print('Skipping file because it does not contain underscore-separated tags: {f}'.format(f=audioFile))
            continue
        if requireNumericalEndTag:
            baseName, index = name.rsplit(sep='_', maxsplit=1)
            try:
                int(index)
            except ValueError:
                print('Skipping file because end tag is not numerical: {f}'.format(f=audioFile))
                continue
        baseName = name
        print(audioFile)

        # Get number of channels in file
        af = wave.open(str(audioFile), 'r')
        nChannels = af.getnchannels()
        af.close()

        if nChannels < 2:
            print('Skipping file because it doesn\'t have multiple channels: {f}'.format(f=audioFile))

        # Increment input file counter
        fileCount += 1

        # Construct ffmpeg command to mix stereo tracks into a mono track. This will be labeled "chanN" where N is after the last split channel
        outName = '{baseName}_chan{k}.wav'.format(k=nChannels, baseName=baseName)
        outFile = audioFile.parents[0] / outName
        ffmpegMixCommand = 'ffmpeg -i "{inFile}" -ac 1 "{outFile}"'.format(inFile=audioFile, outFile=outFile)

        # Run ffmpeg split command
        ffmpegProc = subprocess.call(ffmpegMixCommand)

        # Construct ffmpeg command to split stereo track into two mono tracks. These mono files will be labeled "chan1", "chan2", etc
        mapCommand = []
        for k in range(nChannels):
            outName = '{baseName}_chan{k}.wav'.format(k=k, baseName=baseName, idx=index)
            outFile = audioFile.parents[0] / outName
            mapCommand.append('-map_channel 0.0.{k} "{outFile}"'.format(k=k, outFile=outFile))
        mapCommand = ' '.join(mapCommand)
        ffmpegSplitCommand = 'ffmpeg -i "{inFile}" {mapCommand}'.format(inFile=audioFile, mapCommand=mapCommand)
        # Run ffmpeg split command
        ffmpegProc = subprocess.call(ffmpegSplitCommand)

        # Increment split file counter
        splitCount += nChannels

    print()
    print('Result:')
    print('Split {n} multi-channel wav files into {m} mono wav files.'.format(n=fileCount, m=splitCount))

if __name__ == "__main__":
    # A utility to split multi-channel audio wav files within a folder into individual wav files
    folderPaths = [Path(p) for p in sys.argv[1:]]
    splitAudioTracksInFolder(folderPaths, dryRun=False)

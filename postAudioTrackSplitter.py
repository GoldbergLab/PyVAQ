import sys
from pathlib import Path
import wave
import shutil
import subprocess
from PostProcessingUtilities import loadChannelConfiguration

FFMPEG_EXE = shutil.which('ffmpeg')

# A script to split mutli-channel audio files into separate single-channel audio
#   files, plus one file that has both stereo tracks mixed into one mono track

def splitAudioTracksInFolder(folderPaths, channelNumbers=[], channelNames=[], channelFolders=[], overwrite=False, dryRun=False, requireNumericalEndTag=True, signalPresentTag='A'):
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
        name = audioFile.stem
        folder = audioFile.parent

        # Check that file matches expected filename pattern
        if '_' not in name:
            print('Skipping file because it does not contain underscore-separated tags: {f}'.format(f=audioFile))
            continue

        baseName, index = name.rsplit(sep='_', maxsplit=1)

        try:
            index = int(index)
            outNamePattern = '{baseName}_{index}_{{channelName}}{{signalPresentTag}}{ext}'.format(baseName=baseName, index=index, ext=extension)
        except ValueError:
            # No numerical end tag found
            if requireNumericalEndTag:
                print('Skipping file because end tag is not numerical: {f}'.format(f=audioFile))
                continue
            else:
                index = None
                baseName = name
                outNamePattern = '{baseName}_{{channelName}}{{signalPresentTag}}{ext}'.format(baseName=baseName, index=index, ext=extension)

        print('Splitting audio file {f}'.format(f=audioFile))

        # Get number of channels in file
        af = wave.open(str(audioFile), 'r')
        nChannels = af.getnchannels()
        af.close()

        if nChannels < 2:
            print('Skipping file because it doesn\'t have multiple channels: {f}'.format(f=audioFile))

        # Increment input file counter
        fileCount += 1

        # # Construct ffmpeg command to mix stereo tracks into a mono track. This will be labeled "chan0"
        # outName = '{baseName}_chan{k}.wav'.format(k=0, baseName=baseName)
        # outFile = audioFile.parents[0] / outName
        # ffmpegMixCommand = 'ffmpeg -i "{inFile}" -ac 1 "{outFile}"'.format(inFile=audioFile, outFile=outFile)
        #
        # # Run ffmpeg split command
        # ffmpegProc = subprocess.call(ffmpegMixCommand)

        # Construct ffmpeg command to split stereo track into two mono tracks. These mono files will be labeled "chan1", "chan2", etc
        mapCommand = []
        for channelIndex in range(nChannels):
            if len(channelNumbers) > 0:
                channelNumber = channelNumbers[channelIndex]
            else:
                channelNumber = channelIndex
            if len(channelNames) > 0:
                channelName = channelNames[channelIndex]
            else:
                channelName = 'chan{k}'.format(k=channelindex)
            if len(channelFolders) > 0:
                channelFolder = channelFolders[channelIndex]
            else:
                channelFolder = None

            # In the future this is where audio thresholding will happen
            spt = ''

            # Create the output file name
            outName = outNamePattern.format(index=channelIndex+1,
                baseName=baseName, idx=index, channelName=channelName,
                signalPresentTag=spt)

            # Determine the full output directory
            if channelFolder is not None:
                if channelFolder.is_absolute():
                    outDir = channelFolder
                else:
                    outDir = folder / channelFolder
            else:
                outDir = folder

            # Create the output folder if it doesn't already exist
            outDir.mkdir(parents=True, exist_ok=True)

            # Determine the full output path
            outPath = outDir / outName

            # Create and add on a map command for this channel
            mapCommand.append('-map_channel 0.0.{k} "{outPath}"'.format(
                    k=channelIndex, outPath=outPath)
                )

        mapCommand = ' '.join(mapCommand)
        ffmpegSplitCommand = 'ffmpeg -i "{inFile}" {mapCommand}'.format(
                inFile=audioFile, mapCommand=mapCommand
            )
        # Run ffmpeg split command
        ffmpegProc = subprocess.call(ffmpegSplitCommand)

        # Increment split file counter
        splitCount += nChannels

    print()
    print('Result:')
    print('Split {n} multi-channel wav files into {m} mono wav files.'.format(
            n=fileCount, m=splitCount)
        )

if __name__ == "__main__":
    # A utility to split multi-channel audio wav files within a folder into individual wav files
    sys.argv.pop(0)
    try:
        # Find where '-c' is in the argument list
        configIndex = sys.argv.index('-c')
        # Pop off the '-c'.
        sys.argv.pop(configIndex)
        # Now the config path should be in that index
        configPath = Path(sys.argv.pop(configIndex))
        channelNumbers, channelNames, channelFolders = loadChannelConfiguration(configPath)
    except ValueError:
        channelNumbers, channelNames, channelFolders = ([], [], [])
    folderPaths = [Path(p) for p in sys.argv]

    splitAudioTracksInFolder(folderPaths, channelNumbers=channelNumbers,
        channelNames=channelNames, channelFolders=channelFolders, dryRun=False)

import sys
from pathlib import Path
import shutil
import subprocess
import NCFileUtilities as ncfu
from StateMachineProcesses import extractBooleanDataFromDigitalArray
from PostProcessingUtilities import loadChannelConfiguration

FFMPEG_EXE = shutil.which('ffmpeg')

# A script to split mutli-channel audio files into separate single-channel audio
#   files, plus one file that has both stereo tracks mixed into one mono track

def splitDigitalFilesInFolder(folderPaths, lineNumbers=[], lineNames=[], lineFolders=[], overwrite=False, dryRun=False, requireNumericalEndTag=True, signalPresentTag='P'):
    '''Loop through files in a folder and split multichannel digital .nc files
        into single channel .nc files

    Arguments:
        folderPaths = one or more paths to the folder to look in (either a
            string or a pathlib.Path object)
        lineNumbers = a list of integers corresponding to the NI DAQ line numbers
            for each channel
        lineNames = a list of strings with a name to give the single channel file
            created for each line
        lineFolders = a list of strings with a path to a folder to put each
            line's split data files into. If the folders don't exist, they
            will be created. Relative paths will be relative to the path the
            data file was found in.
        overwrite = (optional) boolean flag indicating whether or not to
            overwrite files if they already exist. Default is False.
        dryRun = (optional) boolean flag indicating whether or not to do a
            dry run, printing the expected behavior rather than doing it.
            Default is False.
        requireNumericalEndTag = (optional) optional boolean flag indicating that
            a numerical end-tag to the filename is required. This will restrict
            the action of the script to audio files with the filename format
            produced by PyVAQ running SimpleDigitalWriter.
        signalPresentTag - a string to tag the filename with if at least one
            digital high detected in the file.
    '''

    if type(folderPaths) != type([]):
        # User passed a single path in - wrap it in a list for consistency.
        folderPaths = [folderPaths]

    files = []
    for folderPath in folderPaths:
        # Compile a list of all matching files
        for k, subpath in enumerate(folderPath.iterdir()):
            if subpath.is_file():
                extension = subpath.suffix.lower()
                if extension == '.nc':
                    files.append(subpath)

    fileCount = 0
    splitCount = 0

    for dataFile in files:
        extension = dataFile.suffix.lower()
        name = dataFile.stem
        folder = dataFile.parent

        # Check that file matches expected filename pattern
        if '_' not in name:
            print('Skipping file because it does not contain underscore-separated tags: {f}'.format(f=dataFile))
            continue

        baseName, index = name.rsplit(sep='_', maxsplit=1)

        try:
            index = int(index)
            outNamePattern = '{baseName}_{index}_{{lineName}}{{signalPresentTag}}.nc'.format(baseName=baseName, index=index)
        except ValueError:
            # No numerical end tag found
            if requireNumericalEndTag:
                print('Skipping file because end tag is not numerical: {f}'.format(f=dataFile))
                continue
            else:
                index = None
                baseName = name
                outNamePattern = '{baseName}_{{lineName}}{{signalPresentTag}}.nc'.format(baseName=baseName, index=index)

        print('Splitting data file {f}'.format(f=dataFile))

        # Load the data from file
        data = ncfu.readNCFile(dataFile)

        if not dryRun:
            # Extract the actual boolean data from the packed integer data
            booleanData = extractBooleanDataFromDigitalArray(data['data'], lineNumbers)

        # Get number of channels in file
        nChannels = len(lineNumbers)

        # Increment input file counter
        fileCount += 1

        if dryRun:
            print('This would have split {p}:'.format(p=str(dataFile)))

        for lineIndex, (lineNumber, lineName, lineFolder) in enumerate(zip(lineNumbers, lineNames, lineFolders)):
            booleanChannelData = booleanData[:, lineIndex]

            if signalPresentTag is not None and len(signalPresentTag) > 0:
                if booleanChannelData.sum() > 0:
                    spt = '_' + signalPresentTag
                else:
                    spt = ''

            # Create the output file name
            outName = outNamePattern.format(baseName=baseName, lineName=lineName, index=index, signalPresentTag=spt)

            # Determine the full output directory
            if lineFolder.is_absolute():
                outDir = lineFolder
            else:
                outDir = folder / lineFolder

            # Create the output folder if it doesn't already exist
            outDir.mkdir(parents=True, exist_ok=True)

            # Determine the full output path
            outPath = outDir / outName

            if dryRun:
                print('  {idx}: Line {num} - {name} ==> {path}'.format(idx=lineIndex, num=lineNumber, name=lineName, path=outPath))
            else:
                # Write the single channel of data to the output file
                metaData = "Digital input channel {ln} ({name} perch sensor)".format(ln=lineNumber, name=lineName)
                ncfu.writeNCFile(outPath, data['time'], data['dt'], lineNumber, metaData, booleanChannelData, dataType='i1')

        # Increment split file counter
        splitCount += nChannels

    print()
    print('Result:')
    print('Split {n} multi-channel wav files into {m} mono wav files.'.format(n=fileCount, m=splitCount))

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
        lineNumbers, lineNames, lineFolders = loadChannelConfiguration(configPath)
    except ValueError:
        lineNumbers, lineNames, lineFolders = ([], [], [])
    folderPaths = [Path(p) for p in sys.argv]

    splitDigitalFilesInFolder(folderPaths, lineNumbers=lineNumbers,
        lineNames=lineNames, lineFolders=lineFolders, dryRun=False)

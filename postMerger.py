import sys
import os
from pathlib import Path
import os
import math

# A script to merge audio/video files in a directory based on their numbering.
#   Files must be NAMEA_XYZ.avi or NAMEB_XYZ.wav, where NAMEA and NAMEB are any
#   set of characters, and XYZ is a three digit number that can be used to match
#   corresponding files. All files in a series must share the same NAME.
#
# This is the format used by PyVAQ running SimpleVideoWriter and
#   SimpleAudioWriter

def mergeFolder(folderPaths, overwrite=False, dryRun=False):
    '''Loop through files in a folder and merge matching audio/video files

    Arguments:
        folderPaths = one or more paths to the folder to look in (either a
            string or a pathlib.Path object)
        overwrite = (optional) boolean flag indicating whether or not to
            overwrite files if they already exist. Default is False.
        dryRun = (optional) boolean flag indicating whether or not to do a
            dry run, printing the expected behavior rather than doing it.
            Default is False.
    '''

    if type(folderPaths) != type([]):
        # User passed a single path in - wrap it in a list for consistency.
        folderPaths = [folderPaths]

    videoStreams = {}
    audioStreams = {}
    for folderPath in folderPaths:
        for k, subpath in enumerate(folderPath.iterdir()):
            if subpath.is_file():
                extension = subpath.suffix.lower()
                name = subpath.name[:-len(extension)]
                if '_' not in name:
                    print('Skipping file because it does not contain underscore-separated tags: {f}'.format(f=subpath))
                    continue
                baseName, index = name.rsplit(sep='_', maxsplit=1)
                try:
                    index = int(index)
                except ValueError:
                    print('Skipping file because end tag is not numerical: {f}'.format(f=subpath))
                    continue
                if extension == '.avi':
                    if baseName not in videoStreams:
                        videoStreams[baseName] = {}
                    if index in videoStreams[baseName]:
                        print('Warning, duplicate video files found for baseName {b} and index {i}'.format(b=baseName, i=index))
                    videoStreams[baseName][index] = subpath
                elif extension == '.wav':
                    if baseName not in audioStreams:
                        audioStreams[baseName] = {}
                    if index in audioStreams[baseName]:
                        print('Warning, duplicate audio files found for baseName {b} and index {i}'.format(b=baseName, i=index))
                    audioStreams[baseName][index] = subpath

    print('Found {n} video streams'.format(n=len(videoStreams)))
    for baseName in videoStreams:
        print('\tStream {b} - {n} video files'.format(b=baseName, n=len(videoStreams[baseName])))
    print('Found {n} audio streams'.format(n=len(audioStreams)))
    for baseName in audioStreams:
        print('\tStream {b} - {n} audio files'.format(b=baseName, n=len(audioStreams[baseName])))

    if len(audioStreams) > 1:
        raise ValueError('Error, found {n} audio streams, cannot handle more than one audio stream.'.format(n=len(audioStreams)))
    if len(audioStreams) < 1:
        raise ValueError('Error, did not find any audio streams.')
    if len(videoStreams) < 1:
        raise ValueError('Error, did not find any video streams.')

    audioStream = list(audioStreams.values())[0]

    unMergedVideoFiles = []
    unMergedAudioFiles = []

    for baseName in videoStreams:
        # Scan for audio files that don't have a counterpart in this video stream.
        for index in audioStream:
            if index not in videoStreams[baseName]:
                unMergedAudioFiles.append(audioStream[index])
                print('Skipping Index {i} found in audio stream; not found in video stream {b}.'.format(i=index, b=baseName))
        for index in videoStreams[baseName]:
            if index not in audioStream:
                unmergedVideoFiles.append(videoStreams[baseName][index])
                print('Skipping Index {i} found in {b} video stream; not found in audio stream.'.format(i=index, b=baseName))
                continue
            mergeFiles(videoFiles=[videoStreams[baseName][index]], audioFile=audioStream[index], dryRun=dryRun, overwrite=overwrite)

def mergeFiles(videoFiles, audioFile, reencodeVideo=False, reencodeAudio=False, compresion=None, overwrite=False, dryRun=False):
    if len(videoFiles) == 1:
        # Construct command template
        mergeCommandTemplate = 'ffmpeg -i "{videoFile}" -i "{audioFile}" -c:a "{audioCodec}" {audioCodecArgs} -c:v "{videoCodec}" {videoCodecArgs} -shortest -nostdin -y "{outputFile}"'
        if reencodeVideo:
            videoCodec='libx264'
            videoCodecArgs = '-crf {compression}'.format(compression=compression)
        else:
            videoCodec = 'copy'
            videoCodecArgs = ''
        if reencodeAudio:
            raise ValueError('Audio reencoding not currently supported.')
        else:
            audioCodec = 'copy'
            audioCodecArgs = ''
        videoFile = Path(videoFiles[0])
        videoFileBase = videoFile.name[:-len(videoFile.suffix)]
        outputFile = videoFile.parents[0] / (videoFileBase + '_merged' + videoFile.suffix)

        if not overwrite:
            if outputFile.exists():
                if dryRun:
                    print('Dry run: ')
                print('Failed to merge into {f} because it already exists.'.format(f=outputFile))
                return

        mergeCommand = mergeCommandTemplate.format(videoFile=videoFile,
            audioFile=audioFile, audioCodec=audioCodec,
            audioCodecArgs=audioCodecArgs, videoCodec=videoCodec,
            videoCodecArgs=videoCodecArgs, outputFile=outputFile)
        if dryRun:
            print('Dry run - would have run:')
            print(mergeCommand)
        else:
            status = os.system(mergeCommand)

if __name__ == "__main__":
    # A utility to make all files and folders within a tree as short a name as possible.
    folderPaths = [Path(p) for p in sys.argv[1:]]
    mergeFolder(folderPaths, dryRun=False)

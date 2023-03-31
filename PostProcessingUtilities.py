from pathlib import Path

def loadChannelConfiguration(configFile):
    lineNumbers = []
    lineNames = []
    lineFolders = []
    with open(configFile, 'r') as f:
        for k, line in enumerate(f):
            if k == 0:
                # Skip header line
                continue
            lineNumber, lineName, lineFolder = line.strip().split(maxsplit=2)
            lineNumbers.append(int(lineNumber))
            lineNames.append(lineName)
            lineFolders.append(Path(lineFolder))
    return lineNumbers, lineNames, lineFolders

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
            configItems = line.strip().split()
            lineNumber, lineName, lineFolder = configItems[0:3]
            lineNumbers.append(int(lineNumber))
            lineNames.append(lineName)
            lineFolders.append(Path(lineFolder))
    return lineNumbers, lineNames, lineFolders

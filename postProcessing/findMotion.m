function roiTraces = findMotion(videoPath, ROIs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% findMotion: Identify motion in ROIs in a video
% usage:  [<output args>] = <function name>(<input args>)
%
% where,
%    videoPath is a char array specifying a path to a video file
%    ROIs is a cell array of ROIs, where each ROI is specified by a 2D 
%       array of the form [[x1, x2]; [y1, y2]];
%    <argN> is <description>
%
% <long description>
%
% See also: <related functions>
%
% Version: <version>
% Author:  Brian Kardon
% Email:   bmk27=cornell*org, brian*kardon=google*com
% Real_email = regexprep(Email,{'=','*'},{'@','.'})
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

videoData = loadVideoData(videoPath);

f = figure;
ax = axes(f);
hold(ax, 'on');

roiTraces = {};
for roiNum = 1:length(ROIs)
    roiData = videoData(roi(1,1):roi(1,2), roi(2,1):roi(2,2), :);
    roiTraces{roiNum} = sum(diff(roiData, 1, 3), [1, 2]);
    plot(ax, roiTraces{roiNum});
end

clc; clear all;

caseName = 'syntheticNetwork'; % Set your case directory
segments = dir(fullfile(caseName, 'segment*')); % Get all segment directories

for i = 1:length(segments)
    segmentPath = fullfile(caseName, segments(i).name, 'run');
    timeDirs = dir(segmentPath);
    timeValues = [];
    upstreamQ = [];
    downstreamQ = [];
    
    for j = 1:length(timeDirs)
        if timeDirs(j).isdir && ~startsWith(timeDirs(j).name, '.')
            timeVal = str2double(timeDirs(j).name);
            if isnan(timeVal)
                continue;
            end
            timeValues = [timeValues; timeVal];
            
            Qfile = fullfile(segmentPath, timeDirs(j).name, 'Q.csv');
            if exist(Qfile, 'file')
                Qdata = readmatrix(Qfile);
                if ~isempty(Qdata)
                    upstreamQ = [upstreamQ; Qdata(1)];
                    downstreamQ = [downstreamQ; Qdata(end)];
                else
                    upstreamQ = [upstreamQ; NaN];
                    downstreamQ = [downstreamQ; NaN];
                end
            else
                upstreamQ = [upstreamQ; NaN];
                downstreamQ = [downstreamQ; NaN];
            end
        end
    end
    
    % Sort data based on time values
    [timeValues, sortIdx] = sort(timeValues);
    upstreamQ = upstreamQ(sortIdx);
    downstreamQ = downstreamQ(sortIdx);
    
    % Create a new figure for each segment
    figure;
    hold on;
    plot(timeValues, upstreamQ, '--b', 'DisplayName', 'Upstream Q');
    plot(timeValues, downstreamQ, '-r', 'DisplayName', 'Downstream Q');
    xlabel('Time');
    ylabel('Flow Rate Q');
    title(sprintf('Upstream and Downstream Q vs Time - %s', segments(i).name));
    legend;
    grid on;
    hold off;
end

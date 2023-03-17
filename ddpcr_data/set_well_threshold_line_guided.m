%Pavan Kota

%Set thresholds for each well
saveOverwrite = 0; 
channels = {'HEX (x)', 'FAM (y)'};
allFiles = dir('./raw_data');
thresholds = cell(96,3);
counter = 1;
axF = 14;

gcal = {{'A04', 'A05', 'A06','B01', 'B10', 'B11'}, ...
    {'A07', 'A08', 'A09', 'B03', 'B06'}, ...
    {'A01', 'A02', 'A03', 'B07', 'B09', 'B12'}, ...
    {'A10', 'A11', 'A12', 'B02', 'B04', 'B05', 'B08'}};


allGroupData = cell(4,1); 
for g= 1:4
    figure(g)
    groupData = zeros(0,2); 
    for i = 1:length(gcal{g})
        for j = 1:length(allFiles)
            if contains(allFiles(j).name, 'Amplitude') && contains(allFiles(j).name, gcal{g}{i})
                rawData = readmatrix(['./raw_data/', allFiles(j).name]);        
                %plot(rawData(:,2), rawData(:,1),'bo')
                groupData = [groupData; rawData(:,[2,1])]; 
                break
            end
        end
    end
    %plot(groupData(:,1), groupData(:,2),'o', 'Color', [0.8,0.8,0.8])
    binscatter(groupData(:,1), groupData(:,2), 250)
    colormap('hot')
    colorbar off
    title(['Reference: Group ' num2str(g)], 'FontSize', axF)
    xlabel('HEX', 'FontSize', axF)
    ylabel('FAM', 'FontSize', axF)
    allGroupData(g) = {groupData};
end

for i = 1:length(allFiles)
    %if contains(allFiles(i).name, 'Amplitude')
    if contains(allFiles(i).name, 'D09') % for supp figure
        m2g4ind = strfind(allFiles(i).name, 'M2G4');
        wellID = allFiles(i).name(m2g4ind+5 : m2g4ind+7);
        thresholds(counter, 1) = {wellID};
        
        currentGroup = 0;
        for g = 1:4
            if sum(contains(gcal{g}, wellID))
                currentGroup = g; 
                break
            end
        end

        if currentGroup == 0
            wellNumber = wellID(2:3); 
            if strcmp(wellNumber(1), '0')
                wellNumber = str2double(wellID(3));
            else
                wellNumber = str2double(wellID(2:3));
            end
            currentGroup = mod(wellNumber-1, 4)+1; 
%             if currentGroup == 0
%                 currentGroup = 4; 
%             end

        end

        disp(['Current Group: ' num2str(currentGroup)])

        rawData = readmatrix(['./raw_data/', allFiles(i).name]);        
        
        
        newY = zeros(size(rawData, 1), 2); 
        for j = 1:2
            while 1
                figure
                plot(allGroupData{currentGroup}(:,1), allGroupData{currentGroup}(:,2),'.', 'Color', [0.8, 0.8, 0.8], 'MarkerSize', 4)
                hold on

                plot(rawData(:,2), rawData(:,1),'g.','MarkerSize', 4)

                title([wellID, ': click two points for ', channels{j}, ' threshold'], 'FontSize', axF) 
                xlabel('HEX','FontSize', axF)
                ylabel('FAM','FontSize', axF)
                [xs, ys] = ginput(2);
                m = (ys(1)-ys(2)) / (xs(1) - xs(2)); 
                xline = [min(rawData(:,2)), max(rawData(:,2))]; 
                yline = m*(xline-xs(1))+ys(1);                 
                hold on
                xlfix = xlim;
                ylfix = ylim; 
                plot(xline, yline, 'k-')
                xlim(xlfix)
                ylim(ylfix)
                
                %title(['Threshold line - ', channels{j}])

                if j== 1 % set threshold for HEX channel - if x is greater than the line
                    channelThreshold = rawData(:,2) > (rawData(:,1) - ys(1) + m*xs(1))/m;
                    plot(rawData(channelThreshold,2), rawData(channelThreshold, 1), 'r.', 'MarkerSize', 4)
                    plot(rawData(~channelThreshold,2), rawData(~channelThreshold, 1), 'k.', 'MarkerSize', 4)
                else
                    channelThreshold = rawData(:,1) > m*(rawData(:,2) - xs(1)) + ys(1); 
                    plot(rawData(channelThreshold,2), rawData(channelThreshold, 1), 'r.','MarkerSize', 4)
                    plot(rawData(~channelThreshold,2), rawData(~channelThreshold, 1), 'k.','MarkerSize', 4)
                end
                title('Confirm Selected Threshold')
                xlabel('HEX','FontSize', axF)
                ylabel('FAM','FontSize', axF)
                userCheck = input(['Confirm ' channels{j} ' threshold with 1; any other key will allow reselection: ']);
                
                if userCheck == 1                                                                
                    thresholds(counter, j+1) = {[xs,ys]};
                    newY(:,j) = channelThreshold;
                    close all
                    break                    
                end
                close all
            end
        end

        
        %thresholds(counter, 2) = {input('Enter HEX threshold: ')};
        %thresholds(counter, 3) = {input('Enter FAM threshold: ')};
        
        
        close all
        
        
        
        
        if saveOverwrite
            writematrix(newY, [wellID '_preproc_well_threshold_line_22-07-13.csv']);            
            counter = counter+1; 
        end

    end
end


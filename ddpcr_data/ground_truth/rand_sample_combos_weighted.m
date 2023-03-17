rng(01212021) % rng seed the date of experiment (or experiment prep)

kRange = [2,3,4];
numSamples = [6,6,6];
concOptions = {'conc.#1', 'conc.#2'};

allBact = {'A. baum.', 'B. frag.' , 'E. cloac.', 'E. faec.', 'E. coli', 'K. pneu.', 'P. aeru.', ...
    'S. aur.', 'S. epid.', 'S. sapr.', 'S. agal.', 'S. pneu.'};

bacteria = {'A. baum.', 'B. frag.' , 'E. cloac.', 'E. faec.', 'E. coli', 'K. pneu.', 'P. aeru.', ...
    'Staph.', 'Strep.'};
staph = {'S. aur.', 'S. epid.', 'S. sapr.'};
strep = {'S. agal.', 'S. pneu.'};


C = length(concOptions);
N = length(bacteria);
heatMaps = cell(length(kRange), 1); 
concentrations = cell(length(kRange),1 ); 
allSamples = cell(length(kRange),1);
for k = 1:length(kRange)
    drawTrack = zeros(N,1); 
    expectedRep = (1-nchoosek(N-1,kRange(k))/nchoosek(N,kRange(k))) * numSamples(k); 
    
    hm = zeros(N, numSamples(k)); 
    concDraws = zeros(N, numSamples(k)); 
    kSamples = cell(numSamples(k), kRange(k)+1);
    kSamples(:,1) = num2cell([1:numSamples(k)]');
    for n = 1:numSamples(k)
        % draw distinct species: 
        %nDraw = randsample(N, kRange(k));
        
        %12/9/2020 representation modification
        eqRepWeights = eq_rep(drawTrack, expectedRep); 
        while 1
            nDraw = randsample(N, kRange(k), true, eqRepWeights);
            if length(nDraw) == length(unique(nDraw))
                break
            end
        end
        
        
        hm(nDraw,n) = 1;  
        
        % 12/9/2020 update - weighting draw probability to get some
        % representation of each species at each sparsity level
        drawTrack(nDraw) = drawTrack(nDraw)+1;
        
        for i = 1:kRange(k)
            cDraw = randi(C);  
            concDraws(nDraw(i), n) = cDraw;
            if nDraw(i) < 8 %not staph or strep
                bact = bacteria{nDraw(i)};
            elseif nDraw(i) == 8 %staph
                bact = staph{randi(length(staph))}; 
            elseif nDraw(i) == 9
                bact = strep{randi(length(strep))};
            end
            kSamples(n,i+1) = {[bact ' ' concOptions{cDraw}]};
                    
        end
    end
    heatMaps(k) = {hm};
    concentrations(k) = {concDraws};
    allSamples(k) = {kSamples};
end

for k = 1:length(kRange)
%     figure
%     h = heatmap(heatMaps{k}, 'CellLabelColor', 'None');
%     %h.YDisplayData = bacteria';
%     h.YData = bacteria';
%     colormap('gray')
%     title(['k = ' num2str(kRange(k)) ' samples'])    
    
    % shuffle:     
    hm = heatMaps{k};    
    concDraws = concentrations{k};
    shuffleOrder = 1:6; % num samples
    %shuffleOrder = 1:12;
    %shuffleOrder = [1,5,9,2,6,10,3,7,11,4,8,12];
    %shuffleOrder = [1,5,9,2,6,10,3,7,11,4,8];
    hm = hm(:, shuffleOrder);
    concDraws = concDraws(:, shuffleOrder);
    
    if exist('concs')
        for i = 1:size(hm,2)
            %concSubstitute = zeros(kRange(k), 1);
            concSubstitute = zeros(size(hm,1),1);
            for j = 1:kRange(k)
                sampleString = allSamples{k}{shuffleOrder(i),j+1};
                for n = 1:length(allBact)
                    if contains(sampleString, allBact(n))
                        break
                    end
                end
                concSub = concs(n,str2double(sampleString(end)));
                if n < 8
                    concSubstitute(n) = concSub;
                elseif sum(contains(staph, allBact(n)))
                    concSubstitute(8) = concSub;
                elseif sum(contains(strep, allBact(n)))
                    concSubstitute(9) = concSub;
                end                    
            end
            hm(:,i) = concSubstitute;
        end
    end
    
    figure
    h = heatmap(hm, 'CellLabelColor', 'None');
    h.YData = bacteria';
    colormap('hot');
    title(['k = ' num2str(kRange(k)) ' samples'])    
    heatMaps(k) = {hm};
end


function weights = eq_rep(drawTrack, expectedCount)
epsilon = 1e-3;
% clip anything beyond expectedCount to zero since probs must be > 0
preWeights = max(expectedCount - drawTrack, 0) + epsilon; 
weights = preWeights / sum(preWeights); 
end

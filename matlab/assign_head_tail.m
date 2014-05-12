function [frames,analysis] = assign_head_tail(frames,analysis,bad_frames)

% Assignment parameters
v_threshold = 40; % Average velocity needed for direciton of motion head assignment
o_threshold = 1.3; % how much more often one end must be leading: originally 2
vend_rthreshold = 1.1; % originally 1.2
b_threshold = 0.2;

% Deal with each segment separately
N_segments = size(analysis.segments,1);
analysis.segment_ht_assign_method = -ones(N_segments,1); % By default, did nothing 
for n=1:N_segments
    select = analysis.segments(n,1):analysis.segments(n,2);
    % If the worm is moving enough, assign the head as the end leading
    % movement on average
    if nanmedian(analysis.s(select))>v_threshold
        rdir_ends = analysis.dphi_ends(select,:);
        [~,i_head] = min(abs(rdir_ends),[],2);
        i_head(isnan(rdir_ends(:,1))) = 3;
        
        % Check that one end is leading a significant amount of the time
        if max(sum(i_head==1),sum(i_head==2))/min(sum(i_head==1),sum(i_head==2)) > ...
                o_threshold
            % head is end leading more often
            if sum(i_head==2) > sum(i_head==1) % the skeleton is reversed
                flip(select);
            end
            analysis.segment_ht_assign_method(n) = 1;
            continue;
        end
        % If one end didn't lead often enough, revert to next method
    end
    
    % If the worm isn't moving much, check whether one end is moving more
    % than the other and assign that as the head (foraging movements)
    if max(nanmean(analysis.s_ends(select,:),1)) / ...
            min(nanmean(analysis.s_ends(select,:),1)) > vend_rthreshold
        % If the second end is moving more, the skeleton is reversed
        if diff(nanmean(analysis.s_ends(select,:),1)) > 0
            flip(select);
        end
        analysis.segment_ht_assign_method(n) = 2;
        continue;
    end
    
    % If it's still not clear, resort to looking for the brighter end
    % (sometimes noisy)
    b = measure_brightness(select);
    b = nanmean(b,1);
    if (b(1) - b(2))/b(1) > b_threshold
        % skeleton is correctly oriented
        analysis.segment_ht_assign_method(n) = 3;
    elseif (b(2) - b(1))/(b(2)) > b_threshold
        % skeleton is flipped
        flip(select);
        analysis.segment_ht_assign_method(n) = 3;
    else
        analysis.segment_ht_assign_method(n) = -1; % By default, did nothing 
    end
end

    function flip(selection)
        for idx=1:length(selection)
            frames(selection(idx)).skeleton = flipud(frames(selection(idx)).skeleton);
            frames(selection(idx)).theta = fliplr(frames(selection(idx)).theta);
        end
        analysis.interframe_d(selection,:) = fliplr(analysis.interframe_d(selection,:));
        analysis.ht_orientation(selection,:) = fliplr(analysis.ht_orientation(selection,:));
        analysis.x_ends(selection,:) = [analysis.x_ends(selection,3:4), ...
            analysis.x_ends(selection,1:2)];
        analysis.dphi_ends(selection,:) = fliplr(analysis.dphi_ends(selection,:));
        analysis.v_ends(selection,:) = fliplr(analysis.v_ends(selection,:));
        analysis.s_ends(selection,:) = fliplr(analysis.s_ends(selection,:));
        analysis.phi_ends(selection,:) = fliplr(analysis.phi_ends(selection,:));
    end

    function b_est = measure_brightness(selection)
        parfor i=1:length(selection)
            if bad_frames(selection(i))
                b_est(i,:) = [NaN,NaN];
                continue;
            end
            % Define end regions as 1/6 skeleton length from the end
            L_skel = size(frames(selection(i)).skeleton,1);
            n_end_skel = round(L_skel/6);
            i_end1 = n_end_skel;
            i_end2 = L_skel - n_end_skel;
            % Fit line using local skeleton
            n_line = round(L_skel*0.03);
            d = frames(selection(i)).skeleton(i_end1 + n_line,:) - ...
                frames(selection(i)).skeleton(i_end1 - n_line,:);
            phi1 = atan2(d(:,2),d(:,1));
            d = frames(selection(i)).skeleton(i_end2 + n_line,:) - ...
                frames(selection(i)).skeleton(i_end2 - n_line,:);
            phi2 = atan2(d(:,2),d(:,1));
            % Rotate line 90 +/- 5 degrees
            phi1 = [phi1 + (1/2 - 0.0278)*pi, phi1 + pi/2, phi1 + (1/2 + 0.0278)*pi];
            phi2 = [phi2 + (1/2 - 0.0278)*pi, phi2 + pi/2, phi2 + (1/2 + 0.0278)*pi];
            % Choose shortest line traversing boundary
            d = zeros(2,3);
            for j=1:3
                % End 1
                xy = frames(selection(i)).skeleton(i_end1,:);
                while all(xy > 0) & xy(1) < size(frames(selection(i)).bw_worm,1) & ...
                        xy(2) < size(frames(selection(i)).bw_worm,2) & ...
                        frames(selection(i)).bw_worm(xy(1),xy(2))
                    d(1,j) = d(1,j) + 1;
                    xy = xy + round([cos(phi1(j)),sin(phi1(j))]);
                end
                xy = frames(selection(i)).skeleton(i_end1,:);
                while all(xy > 0) & xy(1) < size(frames(selection(i)).bw_worm,1) & ...
                        xy(2) < size(frames(selection(i)).bw_worm,2) & ...
                        frames(selection(i)).bw_worm(xy(1),xy(2))
                    d(1,j) = d(1,j) + 1;
                    xy = xy - round([cos(phi1(j)),sin(phi1(j))]);
                end

                % End 2
                xy = frames(selection(i)).skeleton(i_end2,:);
                while all(xy > 0) & xy(1) < size(frames(selection(i)).bw_worm,1) & ...
                        xy(2) < size(frames(selection(i)).bw_worm,2) & ...
                        frames(selection(i)).bw_worm(xy(1),xy(2))
                    d(2,j) = d(2,j) + 1;
                    xy = xy + round([cos(phi2(j)),sin(phi2(j))]);
                end
                xy = frames(selection(i)).skeleton(i_end2,:);
                while all(xy > 0) & xy(1) < size(frames(selection(i)).bw_worm,1) & ...
                        xy(2) < size(frames(selection(i)).bw_worm,2) & ...
                        frames(selection(i)).bw_worm(xy(1),xy(2))
                    d(2,j) = d(2,j) + 1;
                    xy = xy - round([cos(phi2(j)),sin(phi2(j))]);
                end
            end
            [~,ix] = min(d,[],2);

            % Separate head and tail by deleting binary pixels on line
            bw = frames(selection(i)).bw_worm;
            xy = frames(selection(i)).skeleton(i_end1,:);
            while all(xy > 0) & xy(1) < size(frames(selection(i)).bw_worm,1) & ...
                    xy(2) < size(frames(selection(i)).bw_worm,2) & ...
                    frames(selection(i)).bw_worm(xy(1),xy(2))
                bw(xy(1),xy(2)) = false;
                xy = xy + round([cos(phi1(ix(1))),sin(phi1(ix(1)))]);
            end
            xy = frames(selection(i)).skeleton(i_end1,:);
            while all(xy > 0) & xy(1) < size(frames(selection(i)).bw_worm,1) & ...
                    xy(2) < size(frames(selection(i)).bw_worm,2) & ...
                    frames(selection(i)).bw_worm(xy(1),xy(2))
                bw(xy(1),xy(2)) = false;
                xy = xy - round([cos(phi1(ix(1))),sin(phi1(ix(1)))]);
            end

            % End 2
            xy = frames(selection(i)).skeleton(i_end2,:);
            while all(xy > 0) & xy(1) < size(frames(selection(i)).bw_worm,1) & ...
                    xy(2) < size(frames(selection(i)).bw_worm,2) & ...
                    frames(selection(i)).bw_worm(xy(1),xy(2))
                bw(xy(1),xy(2)) = false;
                xy = xy + round([cos(phi2(ix(2))),sin(phi2(ix(2)))]);
            end
            xy = frames(selection(i)).skeleton(i_end2,:);
            while all(xy > 0) & xy(1) < size(frames(selection(i)).bw_worm,1) & ...
                    xy(2) < size(frames(selection(i)).bw_worm,2) & ...
                    frames(selection(i)).bw_worm(xy(1),xy(2))
                bw(xy(1),xy(2)) = false;
                xy = xy - round([cos(phi2(ix(2))),sin(phi2(ix(2)))]);
            end

            % Calculate median brightness
            cc = bwconncomp(bw);
            L = labelmatrix(cc);
            px1 = frames(selection(i)).im_worm(L==L(frames(selection(i)).skeleton(1,1), ...
                frames(selection(i)).skeleton(1,2)));
            px2 = frames(selection(i)).im_worm(L==L(frames(selection(i)).skeleton(end,1), ...
                frames(selection(i)).skeleton(end,2)));
            b_est(i,:) = [median(single(px1)),median(single(px2))];
        end
    end

end
function view_irs(ir_source1, ir_source2, rx, fs)
% VIEW_IRS  Interactive viewer for simulated room impulse responses.
%
%   view_irs(ir_source1, ir_source2, rx, fs)
%
%   Use the Prev/Next buttons to step through receivers.
%   The title shows the receiver index and its x,y position in the room.

    current_rx = 1;
    total_rx   = size(ir_source1, 1);

    fig = figure('Position', [100 100 900 550], ...
        'KeyPressFcn', @(~, event) key_handler(event));

    % --- Buttons and index display ---
    uicontrol('Style', 'pushbutton', ...
              'String', '← Prev', ...
              'Position', [20 10 80 30], ...
              'Callback', @(~,~) step(-1));

    uicontrol('Style', 'pushbutton', ...
              'String', 'Next →', ...
              'Position', [110 10 80 30], ...
              'Callback', @(~,~) step(1));

    txt = uicontrol('Style', 'text', ...
                    'Position', [200 10 140 25], ...
                    'String', sprintf('%d / %d', current_rx, total_rx));

    % --- Draw initial plot ---
    update_plot(current_rx);

    % -----------------------------------------------------------------
    function step(dir)
        current_rx = max(1, min(current_rx + dir, total_rx));
        txt.String = sprintf('%d / %d', current_rx, total_rx);
        update_plot(current_rx);
    end

    function update_plot(idx)
        t = (0:size(ir_source1, 2)-1) / fs;
        subplot(2,1,1);
        plot(t, ir_source1(idx,:));
        xlabel('Time (s)'); ylabel('Amplitude');
        title(sprintf('Source 1 → Receiver %d  (x=%.2fm, y=%.2fm)', ...
            idx, rx(idx,1), rx(idx,2)));
        grid on;

        t = (0:size(ir_source2, 2)-1) / fs;
        subplot(2,1,2);
        plot(t, ir_source2(idx,:));
        xlabel('Time (s)'); ylabel('Amplitude');
        title(sprintf('Source 2 → Receiver %d  (x=%.2fm, y=%.2fm)', ...
            idx, rx(idx,1), rx(idx,2)));
        grid on;
    end

    function key_handler(event)
        switch event.Key
            case 'rightarrow'
                step(1);
            case 'leftarrow'
                step(-1);
        end
    end
end
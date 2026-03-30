clear; clc;

load("absorption_coefficients.mat")

room_dim = absorption.room_dim;
S_total = absorption.S_total;
V = absorption.V;
fc = absorption.fc;
alpha_sabine = absorption.alpha_sabine;
alpha_eyring = absorption.alpha_eyring;
T20 = absorption.T20;
RT60 = absorption.RT60;

height = 1.0;
fs = 48000;
sources = [
    6.65, 7.93, 0; % original source 1 from VRLab 
    5.23, 3.49, 0; % original source 2 from VRLab
    0.2, 4.0, 0; % added new sources
    3.5, 0.2, 0;
    6.78, 4.0, 0;
    0.2, 7.5, 0;
    3.5, 7.93, 0;
    0.2, 1.0, 0; 
];

[idxX, idxY] = meshgrid(2:31, 2:31);
rx_x = (idxX - 1) * 0.23;
rx_y = (idxY - 1) * 0.26;
rx_z = height * ones(size(rx_x));
rx = [rx_x(:), rx_y(:), rx_z(:)];

n_sources = size(sources, 1);
ir_all = cell(n_sources, 1);

% estimating impulse responses for each source
for src = 1:n_sources
    fprintf('Simulating RIRs for source %d/%d\n', src, n_sources);
    ir_all{src} = acousticRoomResponse(room_dim, sources(src, :), rx, ...
        Algorithm="image-source", ...
        ImageSourceOrder=5, ...
        SampleRate=fs, ...
        MaterialAbsorption=alpha_sabine, ...
        BandCenterFrequencies=fc);
    fprintf('Done simulating RIRs for source %d.\n', src);
end

% zero padding so all RIRs have the same length
max_len = max(cellfun(@(x) size(x, 2), ir_all));
for src = 1:n_sources
    pad = max_len - size(ir_all{src}, 2);
    ir_all{src} = [ir_all{src}, zeros(size(ir_all{src}, 1), pad)];
end

% save each individual mat file
for src = 1:n_sources
    output_dir = sprintf('individual_RIRs/source_%d', src);
    mkdir(output_dir);
    fprintf("Saving source %d IRs\n", src);

    for i = 1:size(ir_all{src}, 1)
        ImpulseResponse = ir_all{src}(i, :);
        rx_pos = rx(i, :);
        idxX = mod(i - 1, 30) + 2;
        idxY = floor((i - 1) / 30) + 2;
        filename = fullfile(output_dir, sprintf('idxX_%d_idxY_%d.mat', idxX, idxY));
        save(filename, 'ImpulseResponse', 'fs', 'rx_pos');
    end
    fprintf('Saved source %d IRs to "%s"\n', src, output_dir);
end

fprintf("DONE\n");

%view_irs(ir_source1_padded, ir_source2_padded, rx, fs);
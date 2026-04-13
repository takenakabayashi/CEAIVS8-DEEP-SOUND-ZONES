clear; clc;

save_path = 'room_dataset.h5';
if exist(save_path, 'file'), delete(save_path); end
mkdir('simulated_RTFs');

%% parameters
N_rooms   = 20000;
c         = 343;
f_cutoff  = 400;
f_min     = 30;
K         = 40;
freqs     = f_min * (2^(1/12)).^(0:K-1);

x_min = 3.5;  x_max = 10.0;
z_min = 1.5;  z_max = 3.5;
V_min = 50;   V_max = 300;
max_aspect = 3.0;

alpha_set = [0.1, 0.3, 0.6, 0.9];

fx = [1/4, 3/4];
fy = [1/4, 3/4];
fz = [1/3, 2/3];

src_margin = 0.5;
grid_step  = 0.5;
height_vec = 0.8 : grid_step : 1.8;

%% --- pre-sample all room parameters (parfor needs no randomness issues) ---
room_params = cell(N_rooms, 1);
for i = 1:N_rooms
    while true
        Lx = x_min + (x_max - x_min) * rand();
        Lz = z_min + (z_max - z_min) * rand();
        V  = V_min + (V_max - V_min) * rand();
        Ly = V / (Lx * Lz);
        if Ly >= 2.0 && Ly <= 20.0 && max(Lx,Ly)/min(Lx,Ly) <= max_aspect
            break;
        end
    end

    alpha = alpha_set(randi(numel(alpha_set)));

    S_total = 2 * (Lx*Ly + Lx*Lz + Ly*Lz);
    T60 = 0.161 * V / (alpha * S_total);

    room_params{i}.room_dim = [round(Lx,2), round(Ly,2), round(Lz,2)];
    room_params{i}.alpha = alpha;
    room_params{i}.T60 = round(T60, 4);
end

%% --- parallel compute + save to tmp files ---
t_total = tic;

parfor i = 1:N_rooms
    tmp_file = sprintf('simulated_RTFs/room_%05d.mat', i);
    if exist(tmp_file, 'file'), continue; end  % resume if interrupted

    p = room_params{i};
    Lx = p.room_dim(1);
    Ly = p.room_dim(2);
    Lz = p.room_dim(3);
    T60 = p.T60;
    room_dim = p.room_dim;
    alpha = p.alpha;

    % sources
    source_pos = zeros(8, 3);
    src_idx = 1;
    for xi = 1:2
        for yi = 1:2
            for zi = 1:2
                source_pos(src_idx, :) = [Lx*fx(xi), Ly*fy(yi), Lz*fz(zi)];
                src_idx = src_idx + 1;
            end
        end
    end

    % receivers
    rx_vec = src_margin : grid_step : Lx - src_margin;
    ry_vec = src_margin : grid_step : Ly - src_margin;
    [RX, RY, RZ] = meshgrid(rx_vec, ry_vec, height_vec);
    receiver_pos = [RX(:), RY(:), RZ(:)];
    N_rcv = size(receiver_pos, 1);

    % RTFs
    RTF_all = zeros(K, 8, N_rcv, 'like', single(1+1j));
    for s = 1:8
        RTF_all(:, s, :) = greens_function(room_dim, source_pos(s,:), receiver_pos, freqs, T60, c, f_cutoff);
    end

    % save to tmp
    RTF_real     = real(RTF_all);
    RTF_imag     = imag(RTF_all);
    parsave(tmp_file, room_dim, T60, alpha, freqs, source_pos, receiver_pos, RTF_real, RTF_imag);

    fprintf('[Room %d/%d] dim=[%.2f x %.2f x %.2f]m  T60=%.2fs  rcv=%d\n', ...
        i, N_rooms, Lx, Ly, Lz, T60, N_rcv);
end

fprintf('\nAll rooms computed in %s — consolidating into HDF5...\n', format_time(toc(t_total)));

%% --- consolidate tmp files into HDF5 ---
t_save = tic;
for i = 1:N_rooms
    tmp_file = sprintf('simulated_RTFs/room_%05d.mat', i);
    d = load(tmp_file);
    grp = sprintf('/room_%05d', i);

    h5create(save_path, [grp '/room_dim'],     [1 3]);
    h5write( save_path, [grp '/room_dim'],     d.room_dim);

    h5create(save_path, [grp '/T60'],          1);
    h5write( save_path, [grp '/T60'],          d.T60);

    h5create(save_path, [grp '/alpha'],        1);
    h5write( save_path, [grp '/alpha'],        d.alpha);

    h5create(save_path, [grp '/freqs'],        [1 K]);
    h5write( save_path, [grp '/freqs'],        d.freqs);

    h5create(save_path, [grp '/source_pos'],   [8 3]);
    h5write( save_path, [grp '/source_pos'],   d.source_pos);

    N_rcv = size(d.receiver_pos, 1);
    h5create(save_path, [grp '/receiver_pos'], [N_rcv 3]);
    h5write( save_path, [grp '/receiver_pos'], d.receiver_pos);

    h5create(save_path, [grp '/RTF_real'], [K 8 N_rcv], ...
        'Datatype', 'single', 'ChunkSize', [K 1 1], 'Deflate', 4);
    h5write( save_path, [grp '/RTF_real'], d.RTF_real);

    h5create(save_path, [grp '/RTF_imag'], [K 8 N_rcv], ...
        'Datatype', 'single', 'ChunkSize', [K 1 1], 'Deflate', 4);
    h5write( save_path, [grp '/RTF_imag'], d.RTF_imag);

    if mod(i, 500) == 0
        fprintf('  Saved %d/%d rooms (%s elapsed)\n', i, N_rooms, format_time(toc(t_save)));
    end
end

fprintf('HDF5 consolidation done in %s\n', format_time(toc(t_save)));
fprintf('Total time: %s\n', format_time(toc(t_total)));

% optional: clean up tmp files
% rmdir('tmp', 's');

%% =========================================================
function G = greens_function(room_dim, src, rcv_all, freqs, T60, c, f_cutoff)
lx = room_dim(1); ly = room_dim(2); lz = room_dim(3);
V  = lx * ly * lz;

omega = 2*pi*freqs(:);   % [K x 1]
N_rcv = size(rcv_all, 1);

nx_max = ceil(f_cutoff * lx / c * 2);
ny_max = ceil(f_cutoff * ly / c * 2);
nz_max = ceil(f_cutoff * lz / c * 2);

% all mode combinations at once
[NX, NY, NZ] = ndgrid(0:nx_max, 0:ny_max, 0:nz_max);
NX = NX(:)'; NY = NY(:)'; NZ = NZ(:)';  % [1 x N_modes]

% filter to modes below f_cutoff
omega_N = c * pi * sqrt((NX/lx).^2 + (NY/ly).^2 + (NZ/lz).^2);
valid   = omega_N <= (2*pi*f_cutoff);
NX = NX(valid); NY = NY(valid); NZ = NZ(valid); omega_N = omega_N(valid);  % [1 x M]

% normalisation [1 x M]
Lambda2 = (1 + (NX > 0)) .* (1 + (NY > 0)) .* (1 + (NZ > 0)) / V;

% source shape [1 x M]
psi_src = cos(NX*pi*src(1)/lx) .* cos(NY*pi*src(2)/ly) .* cos(NZ*pi*src(3)/lz);

% receiver shape [N_rcv x M]
psi_rcv = cos(NX .* (pi*rcv_all(:,1)/lx)) .* ...
          cos(NY .* (pi*rcv_all(:,2)/ly)) .* ...
          cos(NZ .* (pi*rcv_all(:,3)/lz));

% numerator [N_rcv x M]
numer = (Lambda2 .* psi_src) .* psi_rcv;

% denominator [K x M]
delta = 3 * log(10) / T60;
denom = omega_N.^2 - omega.^2 - 2j*delta.*omega;  % [K x 1] broadcast with [1 x M]

% sum over modes → [K x N_rcv]
G = single(-(  (1./denom) * numer'  ));
end

%% =========================================================
function parsave(fname, room_dim, T60, alpha, freqs, source_pos, receiver_pos, RTF_real, RTF_imag)
% wrapper needed because parfor can't use save() directly
save(fname, 'room_dim', 'T60', 'alpha', 'freqs', 'source_pos', 'receiver_pos', 'RTF_real', 'RTF_imag');
end

%% =========================================================
function s = format_time(seconds)
    h = floor(seconds / 3600);
    m = floor(mod(seconds, 3600) / 60);
    s = floor(mod(seconds, 60));
    s = sprintf('%02d:%02d:%02d', h, m, s);
end